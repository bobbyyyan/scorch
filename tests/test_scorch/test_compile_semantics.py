"""Semantic-preservation regressions for :func:`scorch.compile`."""

import pytest
import torch

import scorch
from scorch import STensor
from scorch import ops as scorch_ops
from scorch.trace import _symbolic_trace

_RENAMED_MATMUL = scorch.matmul


@pytest.fixture(scope="module")
def matmul_inputs():
    """Small deterministic SpMM inputs with non-trivial post-op values."""
    adj_dense = torch.tensor(
        [
            [1.0, 0.0, -0.5, 0.0],
            [0.0, 0.75, 0.0, 0.25],
            [-0.2, 0.0, 0.0, 1.0],
            [0.0, 0.3, 0.4, 0.0],
        ]
    )
    x = torch.tensor(
        [
            [0.2, -0.4, 0.7],
            [0.8, 0.1, -0.3],
            [-0.5, 0.9, 0.6],
            [0.4, -0.2, 0.5],
        ]
    )
    bias = torch.tensor([0.15, -0.25, 0.35])
    adj = STensor.from_csr(adj_dense.to_sparse_csr(), "A")
    return adj, adj_dense, x, bias


def test_compile_preserves_unsupported_op_after_matmul(matmul_inputs):
    """An unsupported consumer must not be dropped from the graph output."""
    adj, adj_dense, x, _ = matmul_inputs

    @scorch.compile
    def compiled(adj, x):
        return torch.sin(scorch.matmul(adj, x, format="dd"))

    expected = torch.sin(adj_dense @ x)
    actual = compiled(adj, x)

    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_compile_executes_graph_after_supported_fusion_prefix(matmul_inputs):
    """Nodes after a fusible add/relu prefix still execute in graph order."""
    adj, adj_dense, x, bias = matmul_inputs

    @scorch.compile
    def compiled(adj, x, bias):
        hidden = scorch.matmul(adj, x, format="dd") + bias
        hidden = torch.relu(hidden)
        return torch.sin(hidden)

    expected = torch.sin(torch.relu(adj_dense @ x + bias))
    actual = compiled(adj, x, bias)

    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_compile_observes_mutated_captured_tensor(matmul_inputs):
    """A copied full graph must retain live captured-tensor identity."""
    adj, adj_dense, x, bias = matmul_inputs
    scale = torch.tensor([1.0, 0.5, -0.25])

    @scorch.compile
    def compiled(adj, x, bias):
        hidden = torch.relu(scorch.matmul(adj, x, format="dd") + bias)
        return hidden * scale

    hidden = torch.relu(adj_dense @ x + bias)
    initial_scale = scale.clone()
    first = compiled(adj, x, bias)

    scale.copy_(torch.tensor([-0.75, 1.25, 2.0]))
    second = compiled(adj, x, bias)

    assert torch.allclose(first, hidden * initial_scale, atol=1e-4, rtol=1e-4)
    assert torch.allclose(second, hidden * scale, atol=1e-4, rtol=1e-4)


def test_compile_preserves_captured_matmul_time_dict(matmul_inputs):
    """Effectful matmul kwargs must eagerly update the original stats dict."""
    adj, adj_dense, x, bias = matmul_inputs
    stats = {}

    @scorch.compile
    def compiled(adj, x, bias):
        hidden = scorch.matmul(adj, x, format="dd", time_dict=stats)
        return hidden + bias

    actual = compiled(adj, x, bias)
    expected = adj_dense @ x + bias

    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
    assert "eval_time" in stats
    assert isinstance(stats["eval_time"], float)
    assert stats["eval_time"] >= 0.0


def test_compile_preserves_matrix_bias_broadcast(matmul_inputs):
    """A matrix bias must not be treated as a native vector-bias epilogue."""
    adj, adj_dense, x, _ = matmul_inputs
    matrix_bias = torch.tensor(
        [
            [0.10, -0.20, 0.30],
            [-0.40, 0.50, -0.60],
            [0.70, -0.80, 0.90],
            [-1.00, 1.10, -1.20],
        ]
    )

    @scorch.compile
    def compiled(adj, x, bias):
        return torch.relu(scorch.matmul(adj, x, format="dd") + bias)

    expected = torch.relu(adj_dense @ x + matrix_bias)
    actual = compiled(adj, x, matrix_bias)

    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.filterwarnings(
    "ignore:`torch.jit.script` is deprecated:DeprecationWarning"
)
def test_cached_native_candidate_preserves_forward_ad_bias(matmul_inputs):
    """A cached native candidate must fall back when a dual tensor arrives."""
    adj, adj_dense, x, bias = matmul_inputs

    @scorch.compile
    def compiled(adj, x, bias):
        return scorch.matmul(adj, x, format="dd") + bias

    compiled(adj, x, bias)

    with torch.autograd.forward_ad.dual_level():
        dual_bias = torch.autograd.forward_ad.make_dual(bias, torch.ones_like(bias))
        actual = compiled(adj, x, dual_bias)
        primal, tangent = torch.autograd.forward_ad.unpack_dual(actual)

    assert torch.allclose(primal, adj_dense @ x + bias, atol=1e-4, rtol=1e-4)
    assert tangent is not None
    assert torch.equal(tangent, torch.ones_like(primal))


def test_cached_native_candidate_preserves_lazy_negative_bias(matmul_inputs):
    """A lazy negative view must not expose its unresolved storage to C++."""
    adj, adj_dense, x, bias = matmul_inputs

    @scorch.compile
    def compiled(adj, x, bias):
        return scorch.matmul(adj, x, format="dd") + bias

    compiled(adj, x, bias)
    negative_bias = torch._neg_view(bias)

    actual = compiled(adj, x, negative_bias)
    expected = adj_dense @ x + negative_bias

    assert negative_bias.is_neg()
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_compile_preserves_branches_from_fusible_value(matmul_inputs):
    """All consumers of a shared intermediate must remain in the graph."""
    adj, adj_dense, x, bias = matmul_inputs

    @scorch.compile
    def compiled(adj, x, bias):
        hidden = scorch.matmul(adj, x, format="dd") + bias
        return torch.relu(hidden) + torch.sigmoid(hidden)

    hidden = adj_dense @ x + bias
    expected = torch.relu(hidden) + torch.sigmoid(hidden)
    actual = compiled(adj, x, bias)

    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_compile_preserves_graph_with_two_matmuls(matmul_inputs):
    """Multiple contractions fall back without dropping either result."""
    adj, adj_dense, x, bias = matmul_inputs
    other_x = x.roll(shifts=1, dims=0) * 0.6 - 0.1

    @scorch.compile
    def compiled(adj, x, other_x, bias):
        first = scorch.matmul(adj, x, format="dd")
        second = scorch.matmul(adj, other_x, format="dd")
        return torch.sin(first) + torch.relu(second + bias)

    expected = torch.sin(adj_dense @ x) + torch.relu(adj_dense @ other_x + bias)
    actual = compiled(adj, x, other_x, bias)

    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_compile_preserves_structured_output_and_reused_value(matmul_inputs):
    """A reused result and every member of a tuple output are preserved."""
    adj, adj_dense, x, bias = matmul_inputs

    @scorch.compile
    def compiled(adj, x, bias):
        hidden = torch.tanh(scorch.matmul(adj, x, format="dd") + bias)
        return hidden, hidden.square(), hidden.sum(dim=1)

    hidden = torch.tanh(adj_dense @ x + bias)
    actual = compiled(adj, x, bias)
    expected = (hidden, hidden.square(), hidden.sum(dim=1))

    assert isinstance(actual, tuple)
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        assert torch.allclose(actual_item, expected_item, atol=1e-4, rtol=1e-4)


def test_compile_falls_back_eagerly_without_fusible_subgraph():
    """A traceable graph without scorch.matmul retains eager semantics."""
    x = torch.tensor([[-0.8, 0.2], [0.4, 1.1]])
    bias = torch.tensor([0.3, -0.5])

    @scorch.compile
    def compiled(x, bias):
        return torch.sin(torch.add(x, bias))

    assert torch.equal(compiled(x, bias), torch.sin(torch.add(x, bias)))


def test_compile_falls_back_when_matmul_inputs_are_not_direct_arguments(
    matmul_inputs,
):
    """An unproven candidate is run eagerly instead of partially compiled."""
    adj, adj_dense, x, _ = matmul_inputs

    @scorch.compile
    def compiled(adj, x):
        transformed = torch.sin(x)
        return torch.relu(scorch.matmul(adj, transformed, format="dd"))

    expected = torch.relu(adj_dense @ torch.sin(x))
    actual = compiled(adj, x)

    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_compile_forwards_keyword_calls_eagerly(matmul_inputs):
    """Keyword invocation retains the wrapped function's exact call semantics."""
    adj, adj_dense, x, bias = matmul_inputs

    @scorch.compile
    def compiled(adj, x, bias, gain=1.0):
        hidden = scorch.matmul(adj, x, format="dd") + bias
        return torch.sin(hidden) * gain

    actual = compiled(adj=adj, x=x, bias=bias, gain=0.25)
    expected = torch.sin(adj_dense @ x + bias) * 0.25

    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_compile_falls_back_after_data_dependent_trace_failure(matmul_inputs):
    """Python control flow that FX cannot trace remains eager on every call."""
    adj, adj_dense, x, _ = matmul_inputs

    @scorch.compile
    def compiled(adj, x, use_sin):
        hidden = scorch.matmul(adj, x, format="dd")
        if use_sin:
            return torch.sin(hidden)
        return torch.cos(hidden)

    assert torch.allclose(
        compiled(adj, x, True), torch.sin(adj_dense @ x), atol=1e-4, rtol=1e-4
    )
    assert torch.allclose(
        compiled(adj, x, False), torch.cos(adj_dense @ x), atol=1e-4, rtol=1e-4
    )


def test_symbolic_trace_does_not_rebind_matmul_globals():
    """Tracing must not expose a temporary matmul wrapper through modules."""
    original_public_matmul = scorch.matmul
    original_ops_matmul = scorch_ops.matmul
    observed = []

    def fn(adj, x):
        observed.append((scorch.matmul, scorch_ops.matmul))
        return scorch.matmul(adj, x, format="dd")

    _symbolic_trace(fn)

    assert observed == [(original_public_matmul, original_ops_matmul)]
    assert scorch.matmul is original_public_matmul
    assert scorch_ops.matmul is original_ops_matmul


def _global_alias_fn(observed):
    def fn(adj, x, bias):
        observed.append(_RENAMED_MATMUL)
        return torch.relu(_RENAMED_MATMUL(adj, x, format="dd") + bias)

    return fn


def _closure_alias_fn(observed):
    matmul_alias = scorch.matmul

    def fn(adj, x, bias):
        observed.append(matmul_alias)
        return torch.relu(matmul_alias(adj, x, format="dd") + bias)

    return fn


@pytest.mark.parametrize(
    "fn_factory", [_global_alias_fn, _closure_alias_fn], ids=["renamed", "closure"]
)
def test_matmul_aliases_trace_without_rebinding(matmul_inputs, fn_factory):
    """Renamed globals and closure aliases remain identical while tracing."""
    adj, adj_dense, x, bias = matmul_inputs
    observed = []
    original_matmul = scorch.matmul
    fn = fn_factory(observed)

    traced = _symbolic_trace(fn)
    targets = [node.target for node in traced.graph.nodes if node.op == "call_function"]

    assert sum(target is scorch_ops.matmul for target in targets) == 1
    assert observed == [original_matmul]
    assert scorch.matmul is original_matmul
    assert scorch_ops.matmul is original_matmul

    observed.clear()
    compiled = scorch.compile(fn)
    actual = compiled(adj, x, bias)
    expected = torch.relu(adj_dense @ x + bias)

    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
    assert observed == [original_matmul]
