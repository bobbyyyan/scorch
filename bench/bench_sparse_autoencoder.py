#!/usr/bin/env python3
"""Benchmark weight-sparse autoencoder inference: Scorch vs PyTorch Sparse vs PyTorch Dense.

Train a dense autoencoder on MNIST/CIFAR-10, apply magnitude pruning to create
sparse weight matrices, then benchmark inference with sparse weights across
frameworks. Demonstrates Scorch's SpMM (sparse weight x dense activation)
performance advantage.

Usage:
    conda run -n scorch python bench/bench_sparse_autoencoder.py train --model small
    conda run -n scorch python bench/bench_sparse_autoencoder.py train --model all
    conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model small
    conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model all --sparsity 0.9 0.95 0.99
    conda run -n scorch python bench/bench_sparse_autoencoder.py bench --model small --frameworks scorch pytorch-dense
"""

from __future__ import annotations

import argparse
import csv
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden_dims: List[int]  # [h1, h2]
    dataset: str            # "mnist" or "cifar10"
    lr: float
    epochs: int
    batch_size: int


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "small": ModelConfig("small", [1024, 512], "mnist", lr=1e-3, epochs=20, batch_size=256),
    "medium": ModelConfig("medium", [2048, 1024], "mnist", lr=1e-3, epochs=20, batch_size=256),
    "large": ModelConfig("large", [4096, 2048], "mnist", lr=1e-3, epochs=20, batch_size=256),
    "xlarge": ModelConfig("xlarge", [4096, 2048], "cifar10", lr=1e-3, epochs=30, batch_size=256),
}

ALL_MODELS = list(MODEL_CONFIGS.keys())

SPARSITY_LEVELS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]

BENCH_BATCH_SIZE = 256

FRAMEWORK_ORDER = ["PyTorch Dense", "PyTorch Sparse", "Scorch"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WEIGHT_DIR = Path(__file__).resolve().parent.parent / "weights"

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
    """Global unstructured magnitude pruning (Han et al. 2015).

    Returns (pruned_state_dict, stats_dict) where stats_dict contains per-layer
    nnz, total, and actual_sparsity.
    """
    # Collect all weight magnitudes
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
        nnz = mask.sum().item()
        stats[key] = {
            "nnz": nnz,
            "total": total,
            "actual_sparsity": 1.0 - nnz / total,
        }

    return pruned, stats


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str) -> torch.utils.data.DataLoader:
    """Load MNIST or CIFAR-10, return a DataLoader of flattened float32 images."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.MNIST(
            root=str(DATA_DIR), train=True, download=True, transform=transform,
        )
    elif name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=str(DATA_DIR), train=True, download=True, transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)


def get_input_dim(dataset_name: str) -> int:
    if dataset_name == "mnist":
        return 784
    elif dataset_name == "cifar10":
        return 3072
    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_test_batch(dataset_name: str, batch_size: int) -> torch.Tensor:
    """Load a batch from the test set for benchmarking."""
    if dataset_name == "mnist":
        transform = transforms.ToTensor()
        dataset = torchvision.datasets.MNIST(
            root=str(DATA_DIR), train=False, download=True, transform=transform,
        )
    elif dataset_name == "cifar10":
        transform = transforms.ToTensor()
        dataset = torchvision.datasets.CIFAR10(
            root=str(DATA_DIR), train=False, download=True, transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    images, _ = next(iter(loader))
    return images.view(images.size(0), -1)  # flatten


# ---------------------------------------------------------------------------
# Per-framework inference runners
# ---------------------------------------------------------------------------

def _dense_autoencoder_forward(
    x: torch.Tensor, state_dict: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Manual forward pass using F.linear (dense weights)."""
    x = F.relu(F.linear(x, state_dict["enc1.weight"], state_dict["enc1.bias"]))
    x = F.relu(F.linear(x, state_dict["enc2.weight"], state_dict["enc2.bias"]))
    x = F.relu(F.linear(x, state_dict["dec1.weight"], state_dict["dec1.bias"]))
    x = torch.sigmoid(F.linear(x, state_dict["dec2.weight"], state_dict["dec2.bias"]))
    return x


def run_pytorch_dense(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    warmup: int,
    repeats: int,
) -> Tuple[torch.Tensor, TimingSummary]:
    """Benchmark with dense weights via F.linear."""
    with torch.no_grad():
        def fn() -> torch.Tensor:
            return _dense_autoencoder_forward(x, state_dict)
        return benchmark_fn(fn, warmup=warmup, repeats=repeats)


def run_pytorch_sparse(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    warmup: int,
    repeats: int,
) -> Tuple[torch.Tensor, TimingSummary]:
    """Benchmark with CSR sparse weights via torch.sparse.mm."""
    # Pre-convert weights to CSR
    weight_keys = [k for k in state_dict if k.endswith(".weight")]
    csr_weights: Dict[str, torch.Tensor] = {}
    for key in weight_keys:
        csr_weights[key] = state_dict[key].to_sparse_csr()

    activations = [F.relu, F.relu, F.relu, torch.sigmoid]
    layer_names = ["enc1", "enc2", "dec1", "dec2"]

    with torch.no_grad():
        def fn() -> torch.Tensor:
            h = x
            for layer_name, act in zip(layer_names, activations):
                w_csr = csr_weights[f"{layer_name}.weight"]
                b = state_dict[f"{layer_name}.bias"]
                # W @ x.T -> (out, batch), transpose back
                h = act(torch.sparse.mm(w_csr, h.T).T + b)
            return h
        return benchmark_fn(fn, warmup=warmup, repeats=repeats)


def run_scorch(
    x: torch.Tensor,
    state_dict: Dict[str, torch.Tensor],
    warmup: int,
    repeats: int,
) -> Tuple[torch.Tensor, TimingSummary]:
    """Benchmark with Scorch STensor CSR weights via scorch.matmul."""
    # Pre-convert weights to Scorch STensors
    weight_keys = [k for k in state_dict if k.endswith(".weight")]
    scorch_weights: Dict[str, STensor] = {}
    for key in weight_keys:
        w_csr = state_dict[key].to_sparse_csr()
        scorch_weights[key] = STensor.from_csr(w_csr, key.replace(".weight", ""))

    activations = [F.relu, F.relu, F.relu, torch.sigmoid]
    layer_names = ["enc1", "enc2", "dec1", "dec2"]

    with torch.no_grad():
        def fn() -> torch.Tensor:
            h = x
            for layer_name, act in zip(layer_names, activations):
                w_st = scorch_weights[f"{layer_name}.weight"]
                b = state_dict[f"{layer_name}.bias"]
                h_t = h.T.contiguous()
                h = act(scorch.matmul(w_st, h_t, format="dd").T + b)
            return h
        return benchmark_fn(fn, warmup=warmup, repeats=repeats)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_autoencoder(config: ModelConfig) -> DenseAutoencoder:
    """Train a dense autoencoder and save weights."""
    input_dim = get_input_dim(config.dataset)
    model = DenseAutoencoder(input_dim, config.hidden_dims)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    loader = load_dataset(config.dataset)
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

        if epoch % 5 == 0 or epoch == config.epochs:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch:3d}/{config.epochs}  loss={avg_loss:.6f}")

    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHT_DIR / f"autoencoder_{config.name}.pt"
    torch.save(model.state_dict(), path)
    print(f"  Weights saved to {path}")
    return model


# ---------------------------------------------------------------------------
# CSV results
# ---------------------------------------------------------------------------

AE_CSV_COLUMNS = [
    "Model", "Sparsity", "Framework", "Median_ms", "Min_ms", "Std_ms",
]


class AEResultsCollector:
    """Collect autoencoder benchmark rows and save incrementally to CSV."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.rows: List[Dict[str, Any]] = []
        self._wrote_header = False

    def append(
        self,
        model: str,
        sparsity: float,
        framework: str,
        timing: TimingSummary,
    ) -> None:
        row = {
            "Model": model,
            "Sparsity": sparsity,
            "Framework": framework,
            "Median_ms": timing.median_ms,
            "Min_ms": timing.min_ms,
            "Std_ms": timing.std_ms,
        }
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
    """Run inference benchmark for a single model across sparsity levels."""
    config = MODEL_CONFIGS[config_name]
    input_dim = get_input_dim(config.dataset)

    # Load weights
    weight_path = WEIGHT_DIR / f"autoencoder_{config_name}.pt"
    if not weight_path.exists():
        print(f"  SKIP: weights not found at {weight_path}. Run 'train --model {config_name}' first.")
        return []

    base_state_dict = torch.load(weight_path, weights_only=True)

    # Load test batch
    x = get_test_batch(config.dataset, BENCH_BATCH_SIZE)

    print(f"\nModel: {config_name}  (hidden={config.hidden_dims}, dataset={config.dataset})")
    print(f"  Input: ({BENCH_BATCH_SIZE}, {input_dim})")

    all_results: List[Dict[str, Any]] = []

    for sparsity in sparsity_levels:
        print(f"\n  Sparsity: {sparsity:.0%}")

        # Prune weights
        pruned_sd, stats = magnitude_prune(base_state_dict, sparsity)
        for key, s in stats.items():
            print(f"    {key}: nnz={s['nnz']:,}/{s['total']:,}  sparsity={s['actual_sparsity']:.2%}")

        reference_output: Optional[torch.Tensor] = None
        results: List[Tuple[str, TimingSummary]] = []

        for fw in FRAMEWORK_ORDER:
            if fw.lower() not in [f.lower() for f in frameworks]:
                continue

            print(f"\n    {fw}:")
            try:
                if fw == "PyTorch Dense":
                    output, timing = run_pytorch_dense(x, pruned_sd, warmup, repeats)
                    reference_output = output
                elif fw == "PyTorch Sparse":
                    output, timing = run_pytorch_sparse(x, pruned_sd, warmup, repeats)
                    if reference_output is not None:
                        check_correctness(output, reference_output, f"{fw} vs Dense")
                elif fw == "Scorch":
                    output, timing = run_scorch(x, pruned_sd, warmup, repeats)
                    if reference_output is not None:
                        check_correctness(output, reference_output, f"{fw} vs Dense")
                else:
                    continue

                results.append((fw, timing))
                print(
                    f"      median={timing.median_ms:.3f} ms  "
                    f"min={timing.min_ms:.3f} ms  "
                    f"std={timing.std_ms:.3f} ms"
                )

                if collector is not None:
                    collector.append(config_name, sparsity, fw, timing)

                all_results.append({
                    "model": config_name,
                    "sparsity": sparsity,
                    "framework": fw,
                    "median_ms": timing.median_ms,
                    "min_ms": timing.min_ms,
                    "std_ms": timing.std_ms,
                })

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"      SKIPPED (OOM: {e})")
                else:
                    print(f"      FAILED: {e}")
            except Exception as e:
                print(f"      FAILED: {e}")

        # Print results table for this sparsity level
        if results:
            _print_results_table(config_name, sparsity, results)

    return all_results


def _print_results_table(
    model_name: str,
    sparsity: float,
    results: List[Tuple[str, TimingSummary]],
) -> None:
    """Print a formatted results table."""
    print(
        f"\n    {'=' * 60}\n"
        f"    {model_name} @ {sparsity:.0%} sparsity\n"
        f"    {'=' * 60}"
    )
    header = f"    {'Framework':<20}| {'Median (ms)':>11} | {'Min (ms)':>9} | {'Std (ms)':>9}"
    print(header)
    print(f"    {'-' * 56}")
    for fw, timing in results:
        print(
            f"    {fw:<20}| {timing.median_ms:>11.3f} | {timing.min_ms:>9.3f} | "
            f"{timing.std_ms:>9.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    csv_path: Path,
    output_path: Path,
    models: Optional[List[str]] = None,
) -> None:
    """Line plot: x = sparsity, y = runtime (ms), one line per framework.

    One subplot column per model config.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if models is not None:
        df = df[df["Model"].isin(models)]

    model_names = df["Model"].unique()
    n_models = len(model_names)
    if n_models == 0:
        print("No data to plot.")
        return

    setup_plot_style()
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 5), squeeze=False)

    fw_colors = {
        "PyTorch Dense": COLORS["PyTorch"],
        "PyTorch Sparse": EXTRA_COLORS[0],
        "Scorch": COLORS["Scorch"],
    }
    fw_markers = {
        "PyTorch Dense": "s",
        "PyTorch Sparse": "^",
        "Scorch": "o",
    }

    for idx, model_name in enumerate(model_names):
        ax = axes[0, idx]
        sub = df[df["Model"] == model_name]

        for fw in FRAMEWORK_ORDER:
            fw_data = sub[sub["Framework"] == fw]
            if fw_data.empty:
                continue
            ax.plot(
                fw_data["Sparsity"],
                fw_data["Median_ms"],
                label=fw,
                color=fw_colors.get(fw, "gray"),
                marker=fw_markers.get(fw, "o"),
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Sparsity")
        ax.set_ylabel("Median Runtime (ms)")
        ax.set_title(model_name)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Weight-Sparse Autoencoder Inference", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weight-sparse autoencoder inference benchmark: Scorch vs PyTorch"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train autoencoder and save weights")
    train_parser.add_argument(
        "--model", type=str, default="small",
        help="Model config name or 'all' (default: small)",
    )

    # --- bench ---
    bench_parser = subparsers.add_parser("bench", help="Benchmark sparse autoencoder inference")
    bench_parser.add_argument(
        "--model", type=str, default="small",
        help="Model config name or 'all' (default: small)",
    )
    bench_parser.add_argument(
        "--sparsity", nargs="+", type=float, default=None,
        help=f"Sparsity levels (default: {SPARSITY_LEVELS})",
    )
    bench_parser.add_argument(
        "--frameworks", nargs="+", default=["pytorch-dense", "pytorch-sparse", "scorch"],
        help="Frameworks to benchmark (default: all)",
    )
    bench_parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations (default: 5)",
    )
    bench_parser.add_argument(
        "--repeats", type=int, default=20, help="Timed iterations (default: 20)",
    )
    bench_parser.add_argument(
        "--output-dir", type=str, default="bench_results",
        help="Directory for CSV and plot output (default: bench_results)",
    )
    bench_parser.add_argument(
        "--format", type=str, default="png", choices=["png", "pdf", "svg"],
        help="Plot output format (default: png)",
    )
    bench_parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip benchmarking; load CSV and regenerate plot",
    )

    args = parser.parse_args()
    suppress_torch_warnings()
    torch.manual_seed(42)

    if args.command == "train":
        models = ALL_MODELS if args.model.lower() == "all" else [args.model]
        for model_name in models:
            if model_name not in MODEL_CONFIGS:
                print(f"Unknown model config: {model_name}")
                continue
            config = MODEL_CONFIGS[model_name]
            print(f"\n{'=' * 60}")
            print(f"Training autoencoder: {model_name} (hidden={config.hidden_dims}, dataset={config.dataset})")
            print(f"{'=' * 60}")
            train_autoencoder(config)

    elif args.command == "bench":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "sparse_autoencoder_results.csv"
        plot_path = output_dir / f"sparse_autoencoder.{args.format}"

        models = ALL_MODELS if args.model.lower() == "all" else [args.model]

        # Normalize framework names: CLI uses "pytorch-dense" -> "PyTorch Dense"
        fw_map = {
            "pytorch-dense": "PyTorch Dense",
            "pytorch-sparse": "PyTorch Sparse",
            "scorch": "Scorch",
        }
        frameworks = [fw_map.get(f.lower(), f) for f in args.frameworks]

        sparsity_levels = args.sparsity if args.sparsity is not None else SPARSITY_LEVELS

        if not args.plot_only:
            # Remove old CSV to start fresh
            if csv_path.exists():
                csv_path.unlink()

            collector = AEResultsCollector(csv_path)

            for model_name in models:
                if model_name not in MODEL_CONFIGS:
                    print(f"Unknown model config: {model_name}")
                    continue
                benchmark_model(
                    model_name,
                    sparsity_levels=sparsity_levels,
                    frameworks=frameworks,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    collector=collector,
                )

        # Plot
        if csv_path.exists():
            plot_results(csv_path, plot_path, models=models)
        else:
            print(f"No CSV found at {csv_path}; nothing to plot.")


if __name__ == "__main__":
    main()
