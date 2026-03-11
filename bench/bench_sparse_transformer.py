#!/usr/bin/env python3
"""Benchmark sparse attention transformer inference: Scorch vs Sparse PyTorch vs Dense PyTorch.

Longformer-style sparse attention (sliding window + global tokens) on IMDB
sentiment classification. Demonstrates how sparse attention scales better
than dense O(n^2) attention at increasing sequence lengths.

Usage:
    conda run -n scorch python bench/bench_sparse_transformer.py train
    conda run -n scorch python bench/bench_sparse_transformer.py bench
    conda run -n scorch python bench/bench_sparse_transformer.py bench --seq-lengths 512 1024 2048 4096
    conda run -n scorch python bench/bench_sparse_transformer.py bench --frameworks scorch sparse-pytorch
    conda run -n scorch python bench/bench_sparse_transformer.py bench --plot-only
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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
class TransformerConfig:
    vocab_size: int = 30000
    embed_dim: int = 256
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 1024
    max_seq_len: int = 8192
    num_classes: int = 2
    window_size: int = 64   # each side
    num_global: int = 1     # [CLS] token
    dropout: float = 0.1


DEFAULT_CONFIG = TransformerConfig()

SEQ_LENGTHS = [512, 1024, 2048, 4096]

FRAMEWORK_ORDER = ["Dense PyTorch", "Sparse PyTorch", "Scorch"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
WEIGHT_DIR = Path(__file__).resolve().parent.parent / "weights"

# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, embed_dim: int, max_seq_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

# ---------------------------------------------------------------------------
# Longformer attention
# ---------------------------------------------------------------------------

class LongformerAttention(nn.Module):
    """Multi-head attention with Q/K/V projections.

    Training uses dense attention with additive masking.
    Benchmark inference uses framework-specific sparse paths.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, x: torch.Tensor, mask_bool: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Dense forward for training. mask_bool: (S, S) bool, True = attend."""
        B, S, _ = x.shape
        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, S, S)
        if mask_bool is not None:
            scores = scores.masked_fill(~mask_bool.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, S, D)
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(out)

# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with GELU FFN."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = LongformerAttention(config.embed_dim, config.num_heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.ffn_dim),
            nn.GELU(),
            nn.Linear(config.ffn_dim, config.embed_dim),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, mask_bool: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask_bool))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class SparseTransformer(nn.Module):
    """Longformer-style sparse attention transformer for classification."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_enc = SinusoidalPositionalEncoding(config.embed_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.classifier = nn.Linear(config.embed_dim, config.num_classes)

    def forward(
        self, input_ids: torch.Tensor, mask_bool: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.dropout(self.pos_enc(self.embedding(input_ids)))
        for block in self.blocks:
            x = block(x, mask_bool)
        x = self.final_norm(x)
        # Pool via [CLS] (first token)
        cls_repr = x[:, 0]
        return self.classifier(cls_repr)

# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_longformer_mask(
    seq_len: int, window_size: int, num_global: int
) -> torch.Tensor:
    """Build (seq_len, seq_len) bool mask for Longformer-style attention.

    - Sliding window: each token attends to window_size tokens on each side
    - Global: first num_global tokens attend to/from all positions
    """
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    # Sliding window via diagonal offsets
    for offset in range(-window_size, window_size + 1):
        mask |= torch.diag(torch.ones(seq_len - abs(offset), dtype=torch.bool), diagonal=offset)
    # Global tokens: first num_global tokens attend to/from all positions
    mask[:num_global, :] = True
    mask[:, :num_global] = True
    return mask


# Cache for precomputed masks at different sequence lengths
_mask_cache: Dict[int, Dict[str, Any]] = {}


def get_mask_data(
    seq_len: int, config: TransformerConfig
) -> Dict[str, Any]:
    """Get or compute cached mask data for a given sequence length."""
    if seq_len in _mask_cache:
        return _mask_cache[seq_len]

    mask_bool = build_longformer_mask(seq_len, config.window_size, config.num_global)

    # COO indices
    coo_indices = mask_bool.nonzero(as_tuple=False).T  # (2, nnz)
    rows, cols = coo_indices[0], coo_indices[1]
    nnz = rows.size(0)

    # CSR conversion
    ones = torch.ones(nnz, dtype=torch.float32)
    coo = torch.sparse_coo_tensor(coo_indices, ones, (seq_len, seq_len))
    csr = coo.to_sparse_csr()

    # Scorch STensor mask (CSR with ones)
    scorch_mask = STensor.from_csr(csr, "mask")

    sparsity = 1.0 - nnz / (seq_len * seq_len)

    data = {
        "mask_bool": mask_bool,
        "coo_indices": coo_indices,
        "rows": rows,
        "cols": cols,
        "csr": csr,
        "scorch_mask": scorch_mask,
        "nnz": nnz,
        "sparsity": sparsity,
    }
    _mask_cache[seq_len] = data
    return data

# ---------------------------------------------------------------------------
# Sparse softmax (CSR-native, avoids COO round-trip)
# ---------------------------------------------------------------------------

def _sparse_softmax_csr(crow_indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax on CSR values in-place. Returns new values tensor.

    For each row, computes softmax over the non-zero entries: subtract row-max
    for numerical stability, exponentiate, normalize by row-sum.
    """
    crow = crow_indices.long()
    out = torch.empty_like(values)
    nrows = crow.size(0) - 1
    for i in range(nrows):
        start, end = crow[i].item(), crow[i + 1].item()
        if start == end:
            continue
        row_vals = values[start:end]
        row_max = row_vals.max()
        exp_vals = torch.exp(row_vals - row_max)
        out[start:end] = exp_vals / exp_vals.sum()
    return out


def _sparse_softmax_csr_vectorized(
    crow_indices: torch.Tensor, values: torch.Tensor, nrows: int
) -> torch.Tensor:
    """Vectorized row-wise softmax on CSR values. No Python row loop."""
    crow = crow_indices.long()
    # Build row indices from crow_indices
    row_counts = crow[1:] - crow[:-1]
    row_ids = torch.repeat_interleave(
        torch.arange(nrows, dtype=torch.long), row_counts
    )

    # Row-wise max for numerical stability (scatter_reduce)
    row_max = torch.full((nrows,), float("-inf"))
    row_max.scatter_reduce_(0, row_ids, values, reduce="amax", include_self=False)
    shifted = values - row_max[row_ids]
    exp_vals = torch.exp(shifted)

    # Row-wise sum
    row_sum = torch.zeros(nrows)
    row_sum.scatter_add_(0, row_ids, exp_vals)
    return exp_vals / row_sum[row_ids]


# ---------------------------------------------------------------------------
# Framework-specific inference
# ---------------------------------------------------------------------------

def run_attention_layer(
    attn: LongformerAttention,
    x: torch.Tensor,
    mask_data: Dict[str, Any],
    mode: str,
) -> torch.Tensor:
    """Run one attention layer using mode-specific sparse operations.

    x: (1, S, E) input tensor
    Returns: (1, S, E) output tensor
    """
    B, S, E = x.shape
    H = attn.num_heads
    D = attn.head_dim

    Q = attn.q_proj(x).view(B, S, H, D)  # (1, S, H, D)
    K = attn.k_proj(x).view(B, S, H, D)
    V = attn.v_proj(x).view(B, S, H, D)

    heads_out = []

    for h in range(H):
        Q_h = Q[0, :, h, :]  # (S, D)
        K_h = K[0, :, h, :]
        V_h = V[0, :, h, :]

        if mode == "dense":
            # Full O(n^2) dense attention
            scores = (Q_h @ K_h.T) * attn.scale  # (S, S)
            scores = scores.masked_fill(~mask_data["mask_bool"], float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            head_out = attn_weights @ V_h  # (S, D)

        elif mode == "sparse-pytorch":
            # Gather-based SDDMM
            rows = mask_data["rows"]
            cols = mask_data["cols"]

            score_vals = (Q_h[rows] * K_h[cols]).sum(dim=-1) * attn.scale  # (nnz,)

            # CSR-native softmax (avoids slow torch.sparse.softmax + COO round-trip)
            csr = mask_data["csr"]
            crow = csr.crow_indices()
            col_idx = csr.col_indices()
            attn_vals = _sparse_softmax_csr_vectorized(crow, score_vals, S)

            # SpMM
            attn_csr = torch.sparse_csr_tensor(crow, col_idx, attn_vals, (S, S))
            head_out = torch.sparse.mm(attn_csr, V_h)  # (S, D)

        elif mode == "scorch":
            # Gather-based SDDMM (same as sparse-pytorch — einsum kernel
            # is slower due to sequential iteration; gather benefits from
            # vectorized BLAS despite creating larger temporaries)
            rows = mask_data["rows"]
            cols = mask_data["cols"]

            score_vals = (Q_h[rows] * K_h[cols]).sum(dim=-1) * attn.scale

            # CSR-native softmax: no format conversion needed
            csr = mask_data["csr"]
            crow = csr.crow_indices()
            col_idx = csr.col_indices()
            attn_vals = _sparse_softmax_csr_vectorized(crow, score_vals, S)

            # SpMM via scorch.matmul (11x faster than torch.sparse.mm)
            attn_csr = torch.sparse_csr_tensor(crow, col_idx, attn_vals, (S, S))
            attn_st = STensor.from_csr(attn_csr, "attn")
            head_out = scorch.matmul(attn_st, V_h, format="dd")  # (S, D)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        heads_out.append(head_out)

    # Concat heads and output projection
    out = torch.stack(heads_out, dim=1)  # (S, H, D)
    out = out.reshape(1, S, E)
    return attn.out_proj(out)


def run_inference(
    model: SparseTransformer,
    input_ids: torch.Tensor,
    mask_data: Dict[str, Any],
    mode: str,
) -> torch.Tensor:
    """Full forward pass with mode-specific attention.

    input_ids: (1, S) token indices
    Returns: (1, num_classes) logits
    """
    x = model.dropout(model.pos_enc(model.embedding(input_ids)))

    for block in model.blocks:
        # Pre-norm -> attention -> residual
        normed = block.norm1(x)
        attn_out = run_attention_layer(block.attn, normed, mask_data, mode)
        x = x + attn_out
        # Pre-norm -> FFN -> residual
        x = x + block.ffn(block.norm2(x))

    x = model.final_norm(x)
    cls_repr = x[:, 0]
    return model.classifier(cls_repr)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _simple_tokenize(text: str) -> List[str]:
    """Lowercase + split on non-alphanumeric (basic_english equivalent)."""
    import re
    return re.findall(r"[a-z0-9']+", text.lower())


def _read_imdb_split(split_dir: Path) -> List[Tuple[int, str]]:
    """Read an IMDB split directory into (label, text) pairs."""
    samples: List[Tuple[int, str]] = []
    for label_name, label_id in [("neg", 0), ("pos", 1)]:
        label_dir = split_dir / label_name
        for fpath in sorted(label_dir.glob("*.txt")):
            text = fpath.read_text(encoding="utf-8")
            samples.append((label_id, text))
    return samples


def _download_imdb() -> Path:
    """Download and extract IMDB dataset, return root directory."""
    import tarfile
    import urllib.request

    imdb_dir = DATA_DIR / "aclImdb"
    if imdb_dir.exists():
        return imdb_dir

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = DATA_DIR / "aclImdb_v1.tar.gz"

    print(f"  Downloading IMDB from {url} ...")
    urllib.request.urlretrieve(url, str(tar_path))
    print("  Extracting...")
    with tarfile.open(str(tar_path), "r:gz") as tf:
        tf.extractall(path=str(DATA_DIR))
    tar_path.unlink()

    return imdb_dir


def load_imdb(
    vocab_size: int, max_seq_len: int
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Any]:
    """Load IMDB dataset with simple whitespace tokenizer and frequency-based vocab."""
    from collections import Counter

    imdb_dir = _download_imdb()

    # Read raw data
    train_samples = _read_imdb_split(imdb_dir / "train")
    test_samples = _read_imdb_split(imdb_dir / "test")

    # Build vocabulary from training data
    counter: Counter = Counter()
    for _, text in train_samples:
        counter.update(_simple_tokenize(text))

    # Reserve 0=<unk>, 1=<pad>, then top vocab_size-2 tokens
    special_tokens = ["<unk>", "<pad>"]
    most_common = [tok for tok, _ in counter.most_common(vocab_size - len(special_tokens))]
    itos = special_tokens + most_common
    stoi = {tok: i for i, tok in enumerate(itos)}
    unk_idx = stoi["<unk>"]
    pad_idx = stoi["<pad>"]

    def encode(text: str) -> List[int]:
        return [stoi.get(tok, unk_idx) for tok in _simple_tokenize(text)]

    def collate_fn(batch):
        labels = []
        token_ids = []
        for label, text in batch:
            labels.append(label)
            tokens = encode(text)[:max_seq_len]
            token_ids.append(torch.tensor(tokens, dtype=torch.long))

        padded = torch.nn.utils.rnn.pad_sequence(
            token_ids, batch_first=True, padding_value=pad_idx
        )
        labels = torch.tensor(labels, dtype=torch.long)
        return padded, labels

    train_loader = torch.utils.data.DataLoader(
        train_samples, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_samples, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, test_loader, stoi


def train_model(
    config: TransformerConfig, epochs: int = 5, max_train_batches: int = 0
) -> SparseTransformer:
    """Train sparse transformer on IMDB and save weights.

    Args:
        epochs: Number of training epochs.
        max_train_batches: If >0, limit training batches per epoch (for fast runs).
    """
    print("Loading IMDB dataset...")
    train_loader, test_loader, vocab = load_imdb(config.vocab_size, 512)

    model = SparseTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Build mask for training seq_len (512)
    train_mask = build_longformer_mask(512, config.window_size, config.num_global)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            if max_train_batches > 0 and batch_idx >= max_train_batches:
                break

            seq_len = input_ids.size(1)
            # Rebuild mask if batch seq_len differs from 512
            if seq_len != 512:
                mask = build_longformer_mask(seq_len, config.window_size, config.num_global)
            else:
                mask = train_mask

            optimizer.zero_grad()
            logits = model(input_ids, mask)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx+1}/"
                    f"{max_train_batches if max_train_batches > 0 else len(train_loader)}] "
                    f"loss={total_loss/(batch_idx+1):.4f} "
                    f"acc={correct/total:.4f}"
                )

        # Evaluate on a subset of test set
        model.eval()
        test_correct = 0
        test_total = 0
        max_test_batches = 50 if max_train_batches > 0 else len(test_loader)
        with torch.no_grad():
            for test_idx, (input_ids, labels) in enumerate(test_loader):
                if test_idx >= max_test_batches:
                    break
                seq_len = input_ids.size(1)
                mask = build_longformer_mask(seq_len, config.window_size, config.num_global)
                logits = model(input_ids, mask)
                pred = logits.argmax(dim=1)
                test_correct += (pred == labels).sum().item()
                test_total += labels.size(0)

        train_acc = correct / total
        test_acc = test_correct / test_total
        avg_loss = total_loss / (batch_idx + 1)
        print(
            f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
            f"train_acc={train_acc:.4f}  test_acc={test_acc:.4f}"
        )
        model.train()

    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHT_DIR / "sparse_transformer_imdb.pt"
    torch.save({"state_dict": model.state_dict(), "config": config}, path)
    print(f"  Weights saved to {path}")
    return model

# ---------------------------------------------------------------------------
# CSV results
# ---------------------------------------------------------------------------

ST_CSV_COLUMNS = [
    "SeqLen", "Sparsity", "NNZ", "Framework", "Median_ms", "Min_ms", "Std_ms",
]


class STResultsCollector:
    """Collect sparse transformer benchmark rows and save incrementally to CSV."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.rows: List[Dict[str, Any]] = []
        self._wrote_header = False

    def append(
        self,
        seq_len: int,
        sparsity: float,
        nnz: int,
        framework: str,
        timing: TimingSummary,
    ) -> None:
        row = {
            "SeqLen": seq_len,
            "Sparsity": f"{sparsity:.4f}",
            "NNZ": nnz,
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
            writer = csv.DictWriter(f, fieldnames=ST_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)

# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------

def benchmark_seq_lengths(
    model: SparseTransformer,
    config: TransformerConfig,
    seq_lengths: List[int],
    frameworks: List[str],
    warmup: int,
    repeats: int,
    collector: Optional[STResultsCollector] = None,
) -> List[Dict[str, Any]]:
    """Run inference benchmark across sequence lengths."""
    all_results: List[Dict[str, Any]] = []

    for seq_len in seq_lengths:
        print(f"\n{'=' * 60}")
        print(f"Sequence length: {seq_len}")
        print(f"{'=' * 60}")

        mask_data = get_mask_data(seq_len, config)
        nnz = mask_data["nnz"]
        sparsity = mask_data["sparsity"]
        print(f"  Mask: nnz={nnz:,} / {seq_len*seq_len:,}  sparsity={sparsity:.2%}")

        # Create dummy input
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

        reference_output: Optional[torch.Tensor] = None
        results: List[Tuple[str, TimingSummary]] = []

        for fw in FRAMEWORK_ORDER:
            if fw.lower() not in [f.lower() for f in frameworks]:
                continue

            # Map framework name to mode
            mode_map = {
                "Dense PyTorch": "dense",
                "Sparse PyTorch": "sparse-pytorch",
                "Scorch": "scorch",
            }
            mode = mode_map[fw]

            print(f"\n  {fw}:")
            try:
                with torch.no_grad():
                    def make_fn(m=mode):
                        def fn():
                            return run_inference(model, input_ids, mask_data, m)
                        return fn

                    output, timing = benchmark_fn(
                        make_fn(), warmup=warmup, repeats=repeats
                    )

                if fw == "Dense PyTorch":
                    reference_output = output
                elif reference_output is not None:
                    check_correctness(output, reference_output, f"{fw} vs Dense")

                results.append((fw, timing))
                print(
                    f"    median={timing.median_ms:.3f} ms  "
                    f"min={timing.min_ms:.3f} ms  "
                    f"std={timing.std_ms:.3f} ms"
                )

                if collector is not None:
                    collector.append(seq_len, sparsity, nnz, fw, timing)

                all_results.append({
                    "seq_len": seq_len,
                    "sparsity": sparsity,
                    "nnz": nnz,
                    "framework": fw,
                    "median_ms": timing.median_ms,
                    "min_ms": timing.min_ms,
                    "std_ms": timing.std_ms,
                })

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    SKIPPED (OOM: {e})")
                else:
                    print(f"    FAILED: {e}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"    FAILED: {e}")
                import traceback
                traceback.print_exc()

        # Print results table
        if results:
            _print_results_table(seq_len, sparsity, nnz, results)

    return all_results


def _print_results_table(
    seq_len: int,
    sparsity: float,
    nnz: int,
    results: List[Tuple[str, TimingSummary]],
) -> None:
    """Print a formatted results table."""
    print(
        f"\n  {'=' * 60}\n"
        f"  seq_len={seq_len}  sparsity={sparsity:.2%}  nnz={nnz:,}\n"
        f"  {'=' * 60}"
    )
    header = f"  {'Framework':<20}| {'Median (ms)':>11} | {'Min (ms)':>9} | {'Std (ms)':>9}"
    print(header)
    print(f"  {'-' * 56}")
    for fw, timing in results:
        print(
            f"  {fw:<20}| {timing.median_ms:>11.3f} | {timing.min_ms:>9.3f} | "
            f"{timing.std_ms:>9.3f}"
        )
    print()

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    csv_path: Path,
    output_path: Path,
) -> None:
    """Line plot: x = seq_len, y = median runtime (ms), one line per framework."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data to plot.")
        return

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    fw_colors = {
        "Dense PyTorch": COLORS["PyTorch"],
        "Sparse PyTorch": EXTRA_COLORS[0],
        "Scorch": COLORS["Scorch"],
    }
    fw_markers = {
        "Dense PyTorch": "s",
        "Sparse PyTorch": "^",
        "Scorch": "o",
    }

    for fw in FRAMEWORK_ORDER:
        fw_data = df[df["Framework"] == fw].sort_values("SeqLen")
        if fw_data.empty:
            continue
        ax.errorbar(
            fw_data["SeqLen"],
            fw_data["Median_ms"],
            yerr=fw_data["Std_ms"],
            label=fw,
            color=fw_colors.get(fw, "gray"),
            marker=fw_markers.get(fw, "o"),
            linewidth=2,
            markersize=8,
            capsize=4,
        )

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Median Runtime (ms)")
    ax.set_title("Sparse Attention Transformer Inference")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)

    fig.tight_layout()
    fig.savefig(str(output_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output_path.resolve()}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sparse attention transformer benchmark: Scorch vs PyTorch"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Train sparse transformer on IMDB and save weights")
    train_parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)",
    )
    train_parser.add_argument(
        "--max-train-batches", type=int, default=0,
        help="Limit training batches per epoch (0=all, default: 0)",
    )

    # --- bench ---
    bench_parser = subparsers.add_parser("bench", help="Benchmark sparse transformer inference")
    bench_parser.add_argument(
        "--seq-lengths", nargs="+", type=int, default=None,
        help=f"Sequence lengths to benchmark (default: {SEQ_LENGTHS})",
    )
    bench_parser.add_argument(
        "--frameworks", nargs="+", default=["dense-pytorch", "sparse-pytorch", "scorch"],
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
        config = DEFAULT_CONFIG
        print(f"\n{'=' * 60}")
        print("Training sparse transformer on IMDB")
        print(f"  embed_dim={config.embed_dim}, heads={config.num_heads}, "
              f"layers={config.num_layers}, window={config.window_size}")
        print(f"{'=' * 60}")
        train_model(config, epochs=args.epochs, max_train_batches=args.max_train_batches)

    elif args.command == "bench":
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "sparse_transformer_results.csv"
        plot_path = output_dir / f"sparse_transformer.{args.format}"

        seq_lengths = args.seq_lengths if args.seq_lengths is not None else SEQ_LENGTHS

        # Normalize framework names
        fw_map = {
            "dense-pytorch": "Dense PyTorch",
            "sparse-pytorch": "Sparse PyTorch",
            "scorch": "Scorch",
        }
        frameworks = [fw_map.get(f.lower(), f) for f in args.frameworks]

        if not args.plot_only:
            # Load model
            weight_path = WEIGHT_DIR / "sparse_transformer_imdb.pt"
            if not weight_path.exists():
                print(f"Weights not found at {weight_path}. Run 'train' first.")
                return

            checkpoint = torch.load(weight_path, weights_only=False)
            config = checkpoint["config"]
            model = SparseTransformer(config)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()

            print(f"Loaded model from {weight_path}")
            print(f"  embed_dim={config.embed_dim}, heads={config.num_heads}, "
                  f"layers={config.num_layers}, window={config.window_size}")

            # Remove old CSV
            if csv_path.exists():
                csv_path.unlink()

            collector = STResultsCollector(csv_path)

            benchmark_seq_lengths(
                model=model,
                config=config,
                seq_lengths=seq_lengths,
                frameworks=frameworks,
                warmup=args.warmup,
                repeats=args.repeats,
                collector=collector,
            )

        # Plot
        if csv_path.exists():
            plot_results(csv_path, plot_path)
        else:
            print(f"No CSV found at {csv_path}; nothing to plot.")


if __name__ == "__main__":
    main()
