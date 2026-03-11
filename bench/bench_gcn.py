#!/usr/bin/env python3
"""Benchmark GCN inference: Scorch vs PyTorch vs PyG vs DGL.

Standard Kipf & Welling 2-layer GCN with symmetric normalization:
    A_hat = D_tilde^{-1/2} (A + I) D_tilde^{-1/2}
    H^(l+1) = sigma(A_hat @ (H^(l) @ W^T) + b)

Usage:
    conda run -n scorch python bench/bench_gcn.py train --dataset cora
    conda run -n scorch python bench/bench_gcn.py bench --dataset cora
    conda run -n scorch python bench/bench_gcn.py bench --dataset all --warmup 5 --repeats 20
    conda run -n scorch python bench/bench_gcn.py bench --dataset cora --frameworks pytorch scorch
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import scorch
from scorch import STensor

from _utils import benchmark_fn, suppress_torch_warnings, TimingSummary

# ---------------------------------------------------------------------------
# Optional framework imports
# ---------------------------------------------------------------------------

HAS_PYG = False
HAS_DGL = False

try:
    import torch_geometric  # noqa: F401
    from torch_geometric.datasets import Planetoid, Reddit
    from torch_geometric.nn import GCNConv

    HAS_PYG = True
except ImportError:
    pass

try:
    import dgl  # noqa: F401
    from dgl.nn import GraphConv as DGLGraphConv

    HAS_DGL = True
except (ImportError, FileNotFoundError, OSError):
    pass

HAS_OGB = False
try:
    from ogb.nodeproppred import PygNodePropPredDataset

    HAS_OGB = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetConfig:
    hidden: int
    lr: float
    weight_decay: float
    epochs: int


DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "cora": DatasetConfig(hidden=16, lr=0.01, weight_decay=5e-4, epochs=200),
    "citeseer": DatasetConfig(hidden=16, lr=0.01, weight_decay=5e-4, epochs=200),
    "pubmed": DatasetConfig(hidden=16, lr=0.01, weight_decay=5e-4, epochs=200),
    "ogbn-arxiv": DatasetConfig(hidden=256, lr=0.01, weight_decay=0, epochs=500),
    "reddit": DatasetConfig(hidden=256, lr=0.01, weight_decay=0, epochs=10),
    "ogbn-products": DatasetConfig(hidden=256, lr=0.01, weight_decay=0, epochs=100),
}

ALL_DATASETS = list(DATASET_CONFIGS.keys())

# Datasets that require PyG for loading
_PYG_LOAD_DATASETS = {"cora", "citeseer", "pubmed", "reddit"}
# Datasets that require OGB for loading
_OGB_LOAD_DATASETS = {"ogbn-arxiv", "ogbn-products"}

# ---------------------------------------------------------------------------
# Graph dataset container
# ---------------------------------------------------------------------------


@dataclass
class GraphDataset:
    """Framework-agnostic graph dataset."""

    name: str
    x: torch.Tensor  # (N, F)
    y: torch.Tensor  # (N,)
    edge_index: torch.Tensor  # (2, E)
    num_nodes: int
    num_features: int
    num_classes: int
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_dataset(name: str) -> GraphDataset:
    """Load a graph dataset, returning a framework-agnostic container."""
    name_lower = name.lower()

    if name_lower in _PYG_LOAD_DATASETS and not HAS_PYG:
        raise RuntimeError(
            f"Dataset '{name}' requires torch_geometric. "
            "Install with: pip install torch_geometric"
        )
    if name_lower in _OGB_LOAD_DATASETS and not HAS_OGB:
        raise RuntimeError(
            f"Dataset '{name}' requires ogb. Install with: pip install ogb"
        )

    if name_lower in ("cora", "citeseer", "pubmed"):
        return _load_planetoid(name_lower)
    elif name_lower == "reddit":
        return _load_reddit()
    elif name_lower in ("ogbn-arxiv", "ogbn-products"):
        return _load_ogb(name_lower)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _load_planetoid(name: str) -> GraphDataset:
    dataset = Planetoid(root=str(DATA_DIR / name), name=name)
    data = dataset[0]
    return GraphDataset(
        name=name,
        x=data.x,
        y=data.y,
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )


def _load_reddit() -> GraphDataset:
    dataset = Reddit(root=str(DATA_DIR / "reddit"))
    data = dataset[0]
    return GraphDataset(
        name="reddit",
        x=data.x,
        y=data.y,
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
    )


def _load_ogb(name: str) -> GraphDataset:
    # PyG data classes need to be allowlisted for torch.load in PyTorch >= 2.6
    try:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        torch.serialization.add_safe_globals(
            [DataEdgeAttr, DataTensorAttr, GlobalStorage]
        )
    except (ImportError, AttributeError):
        pass

    dataset = PygNodePropPredDataset(name=name, root=str(DATA_DIR / name))
    data = dataset[0]
    split_idx = dataset.get_idx_split()

    # Labels are (N, 1) — squeeze to (N,)
    y = data.y.squeeze(-1)

    edge_index = data.edge_index
    # OGB graphs are directed — make undirected by adding reverse edges
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    edge_index = torch.cat([edge_index, rev], dim=1)
    # Remove duplicate edges
    edge_index = torch.unique(edge_index, dim=1)

    num_nodes = data.num_nodes

    # Convert split indices to boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[split_idx["train"]] = True
    val_mask[split_idx["valid"]] = True
    test_mask[split_idx["test"]] = True

    return GraphDataset(
        name=name,
        x=data.x,
        y=y,
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_features=data.x.size(1),
        num_classes=dataset.num_classes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )


# ---------------------------------------------------------------------------
# Normalized adjacency: A_hat = D_tilde^{-1/2} (A+I) D_tilde^{-1/2}
# ---------------------------------------------------------------------------


def compute_normalized_adj(
    edge_index: torch.Tensor, num_nodes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute symmetric normalized adjacency with self-loops.

    Returns:
        adj_csr: PyTorch sparse CSR tensor for A_hat
        edge_index_loops: (2, E') edge index including self-loops
        edge_weight: (E',) normalized edge weights
    """
    # Add self-loops
    loop_index = torch.arange(num_nodes, dtype=edge_index.dtype)
    loop_edge = torch.stack([loop_index, loop_index], dim=0)
    edge_index_loops = torch.cat([edge_index, loop_edge], dim=1)

    # Compute degree of A + I
    row, col = edge_index_loops[0], edge_index_loops[1]
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.scatter_add_(0, row, torch.ones(row.size(0), dtype=torch.float32))

    # D_tilde^{-1/2}
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

    # Normalized edge weights
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # Build CSR adjacency
    num_edges = edge_index_loops.size(1)
    adj_coo = torch.sparse_coo_tensor(
        edge_index_loops.long(), edge_weight, (num_nodes, num_nodes)
    )
    adj_csr = adj_coo.to_sparse_csr()

    return adj_csr, edge_index_loops, edge_weight


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


class TorchGCNLayer(nn.Module):
    """Single GCN layer: transform, aggregate, then add bias.

    Computes: A_hat @ (x @ W^T) + b
    Bias is applied after aggregation to match PyG/DGL convention.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = torch.sparse.mm(adj, x)
        x = x + self.bias
        return x


class TorchGCN(nn.Module):
    """2-layer GCN using PyTorch sparse matmul."""

    def __init__(
        self, in_features: int, hidden: int, out_features: int
    ) -> None:
        super().__init__()
        self.conv1 = TorchGCNLayer(in_features, hidden)
        self.conv2 = TorchGCNLayer(hidden, out_features)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, dropout: float = 0.0
    ) -> torch.Tensor:
        x = self.conv1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv2(x, adj)
        return F.log_softmax(x, dim=1)


def scorch_gcn_forward(
    x: torch.Tensor,
    adj_scorch: STensor,
    weights: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Scorch GCN forward pass (inference only)."""
    # Layer 1: transform, aggregate, bias
    x = F.linear(x, weights["conv1.lin.weight"])
    x = scorch.matmul(adj_scorch, x, format="dd")
    x = x + weights["conv1.bias"]
    x = F.relu(x)
    # Layer 2: transform, aggregate, bias
    x = F.linear(x, weights["conv2.lin.weight"])
    x = scorch.matmul(adj_scorch, x, format="dd")
    x = x + weights["conv2.bias"]
    return F.log_softmax(x, dim=1)


def _get_fused_kernels():
    """Load fused SpMM+bias+ReLU kernels from scorch_ops."""
    import scorch_ops

    return scorch_ops.spmm_csr_bias_relu_float, scorch_ops.spmm_csr_bias_float


def scorch_fused_gcn_forward(
    x: torch.Tensor,
    adj_csr: torch.Tensor,
    weights: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Scorch GCN forward with fused SpMM+bias+ReLU kernels.

    Fuses the SpMM, bias add, and ReLU into a single C++ kernel per layer,
    eliminating 2 extra memory passes over the (N, H) output per layer.
    """
    spmm_bias_relu, spmm_bias = _get_fused_kernels()

    # Extract CSR structure
    crow = adj_csr.crow_indices().to(torch.int32)
    col = adj_csr.col_indices().to(torch.int32)
    val = adj_csr.values()
    N = adj_csr.size(0)
    A_shape = [N, N]
    A_mode_indices = [[], [crow, col]]

    # Layer 1: transform -> fused(SpMM + bias + ReLU)
    x = F.linear(x, weights["conv1.lin.weight"])
    H = x.size(1)
    result = spmm_bias_relu(
        [N, H], A_shape, A_mode_indices, val,
        [N, H], [[], []], x.contiguous().view(-1),
        weights["conv1.bias"],
    )
    x = result.storage.value.view(N, H)

    # Layer 2: transform -> fused(SpMM + bias) (no ReLU before softmax)
    x = F.linear(x, weights["conv2.lin.weight"])
    C = x.size(1)
    result = spmm_bias(
        [N, C], A_shape, A_mode_indices, val,
        [N, C], [[], []], x.contiguous().view(-1),
        weights["conv2.bias"],
    )
    x = result.storage.value.view(N, C)

    return F.log_softmax(x, dim=1)


class PyGGCN(nn.Module):
    """2-layer GCN using PyG GCNConv (normalization done externally)."""

    def __init__(
        self, in_features: int, hidden: int, out_features: int
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(
            in_features, hidden, normalize=False, add_self_loops=False
        )
        self.conv2 = GCNConv(
            hidden, out_features, normalize=False, add_self_loops=False
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


class DGLGCN(nn.Module):
    """2-layer GCN using DGL GraphConv (normalization done externally)."""

    def __init__(
        self, in_features: int, hidden: int, out_features: int
    ) -> None:
        super().__init__()
        self.conv1 = DGLGraphConv(
            in_features, hidden, norm="none", allow_zero_in_degree=True
        )
        self.conv2 = DGLGraphConv(
            hidden, out_features, norm="none", allow_zero_in_degree=True
        )

    def forward(
        self,
        g: Any,
        x: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        x = self.conv1(g, x, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(g, x, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


# ---------------------------------------------------------------------------
# Weight sharing / conversion
# ---------------------------------------------------------------------------

WEIGHT_DIR = Path(__file__).resolve().parent.parent / "weights"


def save_weights(
    model: TorchGCN, dataset_name: str
) -> Path:
    """Save canonical weight format from TorchGCN."""
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHT_DIR / f"gcn_{dataset_name}.pt"
    torch.save(model.state_dict(), path)
    print(f"  Weights saved to {path}")
    return path


def load_canonical_weights(dataset_name: str) -> Dict[str, torch.Tensor]:
    """Load canonical weights (TorchGCN format)."""
    path = WEIGHT_DIR / f"gcn_{dataset_name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Weights not found at {path}. Run 'train --dataset {dataset_name}' first."
        )
    return torch.load(path, weights_only=True)


def weights_to_pyg(canonical: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert canonical weights to PyG GCNConv state dict.

    Canonical format (TorchGCN):
        conv1.lin.weight (H, F)    conv1.bias (H,)
        conv2.lin.weight (C, H)    conv2.bias (C,)
    PyG GCNConv expects the same keys — no conversion needed.
    """
    return dict(canonical)


def weights_to_dgl(canonical: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert canonical weights to DGL GraphConv state dict.

    DGL GraphConv stores weight as (in, out) — transpose from (out, in).
    Key mapping: conv1.lin.weight -> conv1.weight, conv1.bias stays conv1.bias.
    """
    dgl_sd: Dict[str, torch.Tensor] = {}
    for key, val in canonical.items():
        if ".lin.weight" in key:
            new_key = key.replace(".lin.weight", ".weight")
            val = val.t()  # (out, in) -> (in, out)
        else:
            new_key = key
        dgl_sd[new_key] = val
    return dgl_sd


# ---------------------------------------------------------------------------
# Per-framework inference runners
# ---------------------------------------------------------------------------


def _accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    pred = logits[mask].argmax(dim=1)
    return (pred == y[mask]).float().mean().item()


def run_pytorch(
    dataset: GraphDataset,
    adj_csr: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    warmup: int,
    repeats: int,
) -> Tuple[float, TimingSummary]:
    cfg = DATASET_CONFIGS[dataset.name]
    model = TorchGCN(dataset.num_features, cfg.hidden, dataset.num_classes)
    model.load_state_dict(weights)
    model.eval()

    with torch.no_grad():
        def fn() -> torch.Tensor:
            return model(dataset.x, adj_csr)

        logits, timing = benchmark_fn(fn, warmup=warmup, repeats=repeats)

    acc = _accuracy(logits, dataset.y, dataset.test_mask)
    return acc, timing


def run_scorch(
    dataset: GraphDataset,
    adj_csr: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    warmup: int,
    repeats: int,
) -> Tuple[float, TimingSummary]:
    # Convert adj to Scorch STensor before timing
    adj_scorch = STensor.from_csr(adj_csr, "adj")

    with torch.no_grad():
        def fn() -> torch.Tensor:
            return scorch_gcn_forward(dataset.x, adj_scorch, weights)

        logits, timing = benchmark_fn(fn, warmup=warmup, repeats=repeats)

    acc = _accuracy(logits, dataset.y, dataset.test_mask)
    return acc, timing


def run_scorch_fused(
    dataset: GraphDataset,
    adj_csr: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    warmup: int,
    repeats: int,
) -> Tuple[float, TimingSummary]:
    with torch.no_grad():
        def fn() -> torch.Tensor:
            return scorch_fused_gcn_forward(dataset.x, adj_csr, weights)

        logits, timing = benchmark_fn(fn, warmup=warmup, repeats=repeats)

    acc = _accuracy(logits, dataset.y, dataset.test_mask)
    return acc, timing


def run_pyg(
    dataset: GraphDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    warmup: int,
    repeats: int,
) -> Optional[Tuple[float, TimingSummary]]:
    if not HAS_PYG:
        print("  PyG: SKIPPED (torch_geometric not installed)")
        return None

    cfg = DATASET_CONFIGS[dataset.name]
    model = PyGGCN(dataset.num_features, cfg.hidden, dataset.num_classes)
    pyg_weights = weights_to_pyg(weights)
    model.load_state_dict(pyg_weights)
    model.eval()

    with torch.no_grad():
        def fn() -> torch.Tensor:
            return model(dataset.x, edge_index, edge_weight)

        logits, timing = benchmark_fn(fn, warmup=warmup, repeats=repeats)

    acc = _accuracy(logits, dataset.y, dataset.test_mask)
    return acc, timing


def run_dgl(
    dataset: GraphDataset,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    weights: Dict[str, torch.Tensor],
    warmup: int,
    repeats: int,
) -> Optional[Tuple[float, TimingSummary]]:
    if not HAS_DGL:
        print("  DGL: SKIPPED (dgl not installed)")
        return None

    cfg = DATASET_CONFIGS[dataset.name]
    model = DGLGCN(dataset.num_features, cfg.hidden, dataset.num_classes)
    dgl_weights = weights_to_dgl(weights)
    model.load_state_dict(dgl_weights)
    model.eval()

    # Build DGL graph
    src, dst = edge_index[0], edge_index[1]
    g = dgl.graph((src, dst), num_nodes=dataset.num_nodes)

    with torch.no_grad():
        def fn() -> torch.Tensor:
            return model(g, dataset.x, edge_weight)

        logits, timing = benchmark_fn(fn, warmup=warmup, repeats=repeats)

    acc = _accuracy(logits, dataset.y, dataset.test_mask)
    return acc, timing


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_gcn(dataset: GraphDataset) -> TorchGCN:
    """Train a 2-layer GCN and return the model."""
    cfg = DATASET_CONFIGS[dataset.name]
    suppress_torch_warnings()

    adj_csr, _, _ = compute_normalized_adj(dataset.edge_index, dataset.num_nodes)

    model = TorchGCN(dataset.num_features, cfg.hidden, dataset.num_classes)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    model.train()
    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad()
        out = model(dataset.x, adj_csr, dropout=0.5)
        loss = F.nll_loss(out[dataset.train_mask], dataset.y[dataset.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                logits = model(dataset.x, adj_csr)
            val_acc = _accuracy(logits, dataset.y, dataset.val_mask)
            train_acc = _accuracy(logits, dataset.y, dataset.train_mask)
            print(
                f"  Epoch {epoch:4d}/{cfg.epochs}  "
                f"loss={loss.item():.4f}  "
                f"train_acc={train_acc:.4f}  "
                f"val_acc={val_acc:.4f}"
            )
            model.train()

    model.eval()
    with torch.no_grad():
        logits = model(dataset.x, adj_csr)
    test_acc = _accuracy(logits, dataset.y, dataset.test_mask)
    print(f"  Test accuracy: {test_acc:.4f}")

    save_weights(model, dataset.name)
    return model


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

FRAMEWORK_ORDER = ["PyTorch", "Scorch", "Scorch (fused)", "PyG", "DGL"]


def benchmark_dataset(
    dataset_name: str,
    frameworks: List[str],
    warmup: int,
    repeats: int,
) -> None:
    """Run inference benchmark for a single dataset across frameworks."""
    print(f"\nLoading dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"  SKIP: {e}")
        return

    print(
        f"  {dataset.num_nodes:,} nodes, "
        f"{dataset.edge_index.size(1):,} edges, "
        f"{dataset.num_features} features, "
        f"{dataset.num_classes} classes"
    )

    # Load canonical weights
    try:
        weights = load_canonical_weights(dataset_name)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return

    # Precompute normalized adjacency
    adj_csr, edge_index_loops, edge_weight = compute_normalized_adj(
        dataset.edge_index, dataset.num_nodes
    )

    results: List[Tuple[str, float, TimingSummary]] = []
    reference_logits: Optional[torch.Tensor] = None

    for fw in FRAMEWORK_ORDER:
        if fw.lower() not in [f.lower() for f in frameworks]:
            continue

        print(f"\n  {fw}:")
        try:
            if fw == "PyTorch":
                acc, timing = run_pytorch(
                    dataset, adj_csr, weights, warmup, repeats
                )
                # Store reference logits for correctness check
                model = TorchGCN(
                    dataset.num_features,
                    DATASET_CONFIGS[dataset.name].hidden,
                    dataset.num_classes,
                )
                model.load_state_dict(weights)
                model.eval()
                with torch.no_grad():
                    reference_logits = model(dataset.x, adj_csr)
            elif fw == "Scorch":
                acc, timing = run_scorch(
                    dataset, adj_csr, weights, warmup, repeats
                )
                # Correctness check vs PyTorch
                if reference_logits is not None:
                    adj_st = STensor.from_csr(adj_csr, "adj")
                    with torch.no_grad():
                        sc_logits = scorch_gcn_forward(
                            dataset.x, adj_st, weights
                        )
                    if torch.allclose(sc_logits, reference_logits, atol=1e-4):
                        print("    Correctness: PASS (matches PyTorch)")
                    else:
                        max_diff = (
                            (sc_logits - reference_logits).abs().max().item()
                        )
                        print(
                            f"    Correctness: WARN (max_diff={max_diff:.6f})"
                        )
            elif fw == "Scorch (fused)":
                acc, timing = run_scorch_fused(
                    dataset, adj_csr, weights, warmup, repeats
                )
                # Correctness check vs PyTorch
                if reference_logits is not None:
                    with torch.no_grad():
                        fused_logits = scorch_fused_gcn_forward(
                            dataset.x, adj_csr, weights
                        )
                    if torch.allclose(fused_logits, reference_logits, atol=1e-4):
                        print("    Correctness: PASS (matches PyTorch)")
                    else:
                        max_diff = (
                            (fused_logits - reference_logits).abs().max().item()
                        )
                        print(
                            f"    Correctness: WARN (max_diff={max_diff:.6f})"
                        )
            elif fw == "PyG":
                result = run_pyg(
                    dataset,
                    edge_index_loops,
                    edge_weight,
                    weights,
                    warmup,
                    repeats,
                )
                if result is None:
                    continue
                acc, timing = result
                # Correctness check vs PyTorch
                if reference_logits is not None:
                    pyg_model = PyGGCN(
                        dataset.num_features,
                        DATASET_CONFIGS[dataset.name].hidden,
                        dataset.num_classes,
                    )
                    pyg_model.load_state_dict(weights_to_pyg(weights))
                    pyg_model.eval()
                    with torch.no_grad():
                        pyg_logits = pyg_model(
                            dataset.x, edge_index_loops, edge_weight
                        )
                    if torch.allclose(pyg_logits, reference_logits, atol=1e-4):
                        print("    Correctness: PASS (matches PyTorch)")
                    else:
                        max_diff = (
                            (pyg_logits - reference_logits).abs().max().item()
                        )
                        print(
                            f"    Correctness: WARN (max_diff={max_diff:.6f})"
                        )
            elif fw == "DGL":
                result = run_dgl(
                    dataset,
                    edge_index_loops,
                    edge_weight,
                    weights,
                    warmup,
                    repeats,
                )
                if result is None:
                    continue
                acc, timing = result
            else:
                continue

            results.append((fw, acc, timing))
            print(
                f"    median={timing.median_ms:.3f} ms  "
                f"min={timing.min_ms:.3f} ms  "
                f"std={timing.std_ms:.3f} ms  "
                f"acc={acc:.4f}"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"    SKIPPED (OOM: {e})")
            else:
                print(f"    FAILED: {e}")
        except Exception as e:
            print(f"    FAILED: {e}")

    # Print results table
    if results:
        _print_results_table(dataset, results)


def _print_results_table(
    dataset: GraphDataset,
    results: List[Tuple[str, float, TimingSummary]],
) -> None:
    """Print a formatted results table."""
    print(
        f"\n{'=' * 72}\n"
        f"Dataset: {dataset.name} "
        f"({dataset.num_nodes:,} nodes, {dataset.edge_index.size(1):,} edges, "
        f"{dataset.num_features} features, {dataset.num_classes} classes)\n"
        f"{'=' * 72}"
    )
    header = f"{'Framework':<14}| {'Median (ms)':>11} | {'Min (ms)':>9} | {'Std (ms)':>9} | {'Accuracy':>8}"
    print(header)
    print("-" * len(header))
    for fw, acc, timing in results:
        print(
            f"{fw:<14}| {timing.median_ms:>11.3f} | {timing.min_ms:>9.3f} | "
            f"{timing.std_ms:>9.3f} | {acc:>8.4f}"
        )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GCN inference benchmark: Scorch vs PyTorch vs PyG vs DGL"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train GCN and save weights")
    train_parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name or 'all' (default: cora)",
    )

    # --- bench ---
    bench_parser = subparsers.add_parser("bench", help="Benchmark GCN inference")
    bench_parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name or 'all' (default: cora)",
    )
    bench_parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations (default: 5)"
    )
    bench_parser.add_argument(
        "--repeats", type=int, default=20, help="Timed iterations (default: 20)"
    )
    bench_parser.add_argument(
        "--frameworks",
        nargs="+",
        default=["pytorch", "scorch", "scorch (fused)", "pyg", "dgl"],
        help="Frameworks to benchmark (default: all)",
    )

    args = parser.parse_args()
    suppress_torch_warnings()
    torch.manual_seed(42)

    if args.command == "train":
        datasets = ALL_DATASETS if args.dataset.lower() == "all" else [args.dataset]
        for ds_name in datasets:
            print(f"\n{'=' * 60}")
            print(f"Training GCN on {ds_name}")
            print(f"{'=' * 60}")
            try:
                dataset = load_dataset(ds_name)
                train_gcn(dataset)
            except Exception as e:
                print(f"  FAILED: {e}")

    elif args.command == "bench":
        datasets = ALL_DATASETS if args.dataset.lower() == "all" else [args.dataset]
        for ds_name in datasets:
            benchmark_dataset(
                ds_name,
                frameworks=args.frameworks,
                warmup=args.warmup,
                repeats=args.repeats,
            )


if __name__ == "__main__":
    main()
