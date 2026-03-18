"""Tests for @scorch.compile kernel fusion decorator."""
import pytest
import torch

import scorch
from scorch import STensor
from scorch.trace import analyze_fx_graph, FusionSpec, compile as scorch_compile, _symbolic_trace


def _make_csr_and_dense(n=64, k=16, density=0.1, seed=42):
    """Create a CSR STensor and a dense torch.Tensor."""
    torch.manual_seed(seed)
    mask = torch.rand(n, n) < density
    vals = torch.rand(n, n) * mask.float()
    csr = vals.to_sparse_csr()
    adj = STensor.from_csr(csr, "A")
    x = torch.rand(n, k)
    return adj, x


def _make_coo_and_dense(n=64, k=16, density=0.1, seed=42):
    """Create a COO STensor and a dense torch.Tensor."""
    torch.manual_seed(seed)
    mask = torch.rand(n, n) < density
    vals = torch.rand(n, n) * mask.float()
    adj = STensor.from_torch(vals, "A").to_sparse("oo")
    x = torch.rand(n, k)
    return adj, x


# ---- Tracing tests ----


def test_trace_produces_fx_graph():
    """_symbolic_trace captures matmul -> add -> relu."""

    def fn(adj, x, bias):
        h = scorch.matmul(adj, x, format="dd")
        h = h + bias
        return torch.relu(h)

    traced = _symbolic_trace(fn)
    node_targets = [n.target for n in traced.graph.nodes if n.op == "call_function"]
    assert scorch.ops.matmul in node_targets


def test_fusion_analysis():
    """FusionSpec correctly identifies contraction + postops."""

    def fn(adj, x, bias):
        h = scorch.matmul(adj, x, format="dd")
        h = h + bias
        return torch.relu(h)

    traced = _symbolic_trace(fn)
    adj, x = _make_csr_and_dense()
    bias = torch.rand(x.size(1))
    spec = analyze_fx_graph(traced.graph, (adj, x, bias))

    assert len(spec.post_ops.ops) == 2
    assert spec.post_ops.ops[0].kind == "add"
    assert spec.post_ops.ops[1].kind == "relu"
    assert len(spec.extra_arg_indices) == 1  # bias
    assert spec.matmul_arg_indices == [0, 1]


# ---- Prebuilt path tests ----


def test_compile_spmm_bias_relu_csr():
    """CSR x dense + bias + relu matches eager computation."""
    adj, x = _make_csr_and_dense()
    bias = torch.rand(x.size(1))

    @scorch.compile
    def fused(adj, x, bias):
        h = scorch.matmul(adj, x, format="dd")
        h = h + bias
        return torch.relu(h)

    result = fused(adj, x, bias)

    # Compute reference eagerly
    adj_dense = adj.to_torch(in_place=False)
    expected = torch.relu(adj_dense @ x + bias)

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected, atol=1e-4)


def test_compile_spmm_bias_csr():
    """CSR x dense + bias only (no activation)."""
    adj, x = _make_csr_and_dense()
    bias = torch.rand(x.size(1))

    @scorch.compile
    def fused(adj, x, bias):
        h = scorch.matmul(adj, x, format="dd")
        return h + bias

    result = fused(adj, x, bias)

    adj_dense = adj.to_torch(in_place=False)
    expected = adj_dense @ x + bias

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected, atol=1e-4)


def test_compile_matches_unfused():
    """Compiled fused == manual unfused (scorch.matmul + torch ops)."""
    adj, x = _make_csr_and_dense()
    bias = torch.rand(x.size(1))

    @scorch.compile
    def fused(adj, x, bias):
        h = scorch.matmul(adj, x, format="dd")
        h = h + bias
        return torch.relu(h)

    fused_result = fused(adj, x, bias)

    # Unfused
    h = scorch.matmul(adj, x, format="dd")
    unfused_result = torch.relu(h + bias)

    assert torch.allclose(fused_result, unfused_result, atol=1e-4)


# ---- Caching tests ----


def test_cache_hit():
    """Second call skips compilation (same cache key)."""
    adj, x = _make_csr_and_dense()
    bias = torch.rand(x.size(1))

    @scorch.compile
    def fused(adj, x, bias):
        h = scorch.matmul(adj, x, format="dd")
        h = h + bias
        return torch.relu(h)

    result1 = fused(adj, x, bias)
    result2 = fused(adj, x, bias)

    assert len(fused._cache) == 1
    assert torch.allclose(result1, result2, atol=1e-6)


def test_cache_isolation():
    """Different postop chains produce separate cache entries."""
    adj, x = _make_csr_and_dense()
    bias = torch.rand(x.size(1))

    @scorch.compile
    def fused_relu(adj, x, bias):
        h = scorch.matmul(adj, x, format="dd")
        h = h + bias
        return torch.relu(h)

    @scorch.compile
    def fused_no_relu(adj, x, bias):
        h = scorch.matmul(adj, x, format="dd")
        return h + bias

    r1 = fused_relu(adj, x, bias)
    r2 = fused_no_relu(adj, x, bias)

    # They should differ (relu zeros out negatives)
    adj_dense = adj.to_torch(in_place=False)
    expected_relu = torch.relu(adj_dense @ x + bias)
    expected_no_relu = adj_dense @ x + bias

    assert torch.allclose(r1, expected_relu, atol=1e-4)
    assert torch.allclose(r2, expected_no_relu, atol=1e-4)


# ---- JIT path tests (COO to bypass prebuilt) ----


def test_compile_spmm_bias_relu_coo():
    """JIT fused kernel matches eager for COO input."""
    adj, x = _make_coo_and_dense()
    bias = torch.rand(x.size(1))

    @scorch.compile
    def fused(adj, x, bias):
        h = scorch.matmul(adj, x, format="dd")
        h = h + bias
        return torch.relu(h)

    result = fused(adj, x, bias)

    adj_dense = adj.to_torch(in_place=False)
    expected = torch.relu(adj_dense @ x + bias)

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected, atol=1e-4)


def test_compile_relu_only():
    """MATMUL -> RELU without bias."""
    adj, x = _make_csr_and_dense()

    @scorch.compile
    def fused(adj, x):
        h = scorch.matmul(adj, x, format="dd")
        return torch.relu(h)

    result = fused(adj, x)

    adj_dense = adj.to_torch(in_place=False)
    expected = torch.relu(adj_dense @ x)

    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, expected, atol=1e-4)
