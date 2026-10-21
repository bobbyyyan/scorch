"""``scorch.matmul`` with ``format=`` records the same derivative as without it.

A product of 1-D and 2-D operands is an einsum, and its derivative is the einsum's
(autograd.py's einsum_grad path), with ``format=`` naming the result's layout. L16 and
864745d refused any keyword argument (``scorch.matmul has no derivative for ... with
format``), which is why the workload harness's gcn training form (``scorch.matmul(A, x,
format="dd")``, wlprof.py) failed in every arm of the merge's speed check. These tests
hold the tip to torch: values and gradients, float32 and float64, a dense and a sparse
requested layout, the sparse operand on either side, an STensor and a torch CSR operand,
and the harness form's two-layer GCN training step.

float64 results are held to 1e-12 of the elementwise bound (|A| |X| for a product, the
same products of absolute values for each gradient); float32 to 1e-5 of it.
"""

import pytest
import torch
import torch.nn.functional as F

import scorch
from scorch.stensor import STensor

REL = {torch.float32: 1e-5, torch.float64: 1e-12}
DTYPES = [torch.float32, torch.float64]


def _csr(rows, cols, density, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    dense = torch.randn(rows, cols, generator=g, dtype=torch.float64)
    dense[torch.rand(rows, cols, generator=g) >= density] = 0
    dense[1] = 0  # an empty row
    return dense.to(dtype).to_sparse_csr()


def _operand(csr, kind, requires_grad):
    values = csr.values().detach().clone().requires_grad_(requires_grad)
    t = torch.sparse_csr_tensor(
        csr.crow_indices(), csr.col_indices(), values, size=csr.shape
    )
    return (STensor.from_torch(t, "A") if kind == "stensor" else t), values


def _dense_of(y):
    if isinstance(y, STensor):
        y = y.to_torch()
    return y.to_dense() if y.layout != torch.strided else y


def _within(got, ref, bound, rel, what):
    assert got.dtype == ref.dtype, (what, got.dtype, ref.dtype)
    # A NaN or an infinity fails by name: either would make `over` below NaN, and a
    # comparison with NaN must not be what decides.
    assert bool(torch.isfinite(got.detach()).all()), (what, "holds a NaN or an infinity")
    err = (got.detach() - ref.detach()).abs()
    over = float((err - rel * bound).max())
    ratio = float((err / bound.clamp_min(1e-300)).max())
    assert over <= 0.0, (what, ratio, "times the bound; allowed", rel)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_judges_fail_a_planted_nan_an_infinity_and_an_excess(dtype):
    """Both judges' negative controls: a planted NaN, an infinity and an error just
    above the allowance each fail _within and the gcn test's assert_close; the
    reference itself passes both."""
    ref = torch.tensor([1.0, -2.0, 3.0], dtype=dtype)
    bound = ref.abs()
    rel = REL[dtype]
    tol = {
        torch.float32: dict(rtol=1e-4, atol=1e-5),
        torch.float64: dict(rtol=1e-10, atol=1e-12),
    }[dtype]
    _within(ref.clone(), ref, bound, rel, "control")
    torch.testing.assert_close(ref.clone(), ref, **tol)
    for bad in (float("nan"), float("inf"), 1.0 + 1e-3):
        got = ref.clone()
        got[0] = bad
        with pytest.raises(AssertionError):
            _within(got, ref, bound, rel, "control")
        with pytest.raises(AssertionError):
            torch.testing.assert_close(got, ref, **tol)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("fmt", ["dd", "ds"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_stensor_product_with_format_matches_torch(side, fmt, dtype):
    csr = _csr(9, 7, 0.4, seed=1, dtype=dtype)
    g = torch.Generator().manual_seed(2)
    other_shape = (7, 5) if side == "left" else (6, 9)
    other = (
        torch.randn(other_shape, generator=g, dtype=torch.float64)
        .to(dtype)
        .requires_grad_(True)
    )
    a, values = _operand(csr, "stensor", requires_grad=True)
    y = (
        scorch.matmul(a, other, format=fmt)
        if side == "left"
        else scorch.matmul(other, a, format=fmt)
    )
    yd = _dense_of(y)
    G = torch.randn(yd.shape, generator=g, dtype=torch.float64).to(dtype)
    got_values, got_other = torch.autograd.grad(yd, (values, other), G)

    dense = csr.to_dense().requires_grad_(True)
    o = other.detach().clone().requires_grad_(True)
    ref = dense @ o if side == "left" else o @ dense
    ref_dense, ref_other = torch.autograd.grad(ref, (dense, o), G)
    absA, absO, absG = csr.to_dense().abs(), other.detach().abs(), G.abs()
    rel = REL[dtype]
    _within(yd, ref, absA @ absO if side == "left" else absO @ absA, rel, "y")
    rows = torch.repeat_interleave(
        torch.arange(csr.shape[0]), csr.crow_indices()[1:] - csr.crow_indices()[:-1]
    )
    cols = csr.col_indices()
    bound_a = absG @ absO.t() if side == "left" else absO.t() @ absG
    _within(got_values, ref_dense[rows, cols], bound_a[rows, cols], rel, "values")
    bound_o = absA.t() @ absG if side == "left" else absG @ absA.t()
    _within(got_other, ref_other, bound_o, rel, "other")


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_torch_csr_product_with_format_matches_torch(side, dtype):
    csr = _csr(9, 7, 0.4, seed=3, dtype=dtype)
    g = torch.Generator().manual_seed(4)
    other_shape = (7, 5) if side == "left" else (6, 9)
    other = (
        torch.randn(other_shape, generator=g, dtype=torch.float64)
        .to(dtype)
        .requires_grad_(True)
    )
    a, _ = _operand(csr, "torch", requires_grad=False)
    y = (
        scorch.matmul(a, other, format="dd")
        if side == "left"
        else scorch.matmul(other, a, format="dd")
    )
    yd = _dense_of(y)
    G = torch.randn(yd.shape, generator=g, dtype=torch.float64).to(dtype)
    (got_other,) = torch.autograd.grad(yd, (other,), G)
    dense = csr.to_dense()
    o = other.detach().clone().requires_grad_(True)
    ref = dense @ o if side == "left" else o @ dense
    (ref_other,) = torch.autograd.grad(ref, (o,), G)
    absA, absO, absG = dense.abs(), other.detach().abs(), G.abs()
    rel = REL[dtype]
    _within(yd, ref, absA @ absO if side == "left" else absO @ absA, rel, "y")
    _within(
        got_other,
        ref_other,
        absA.t() @ absG if side == "left" else absG @ absA.t(),
        rel,
        "other",
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_harness_form_gcn_training_step_matches_torch(dtype):
    """wlprof.py's gcn training step in its harness form, on a small graph: the
    adjacency is a CSR STensor that wants no gradient, and every parameter's gradient
    flows through scorch.matmul(A, x, format="dd")."""
    n, f_in, hidden, classes = 40, 12, 8, 3
    adj = _csr(n, n, 0.15, seed=5, dtype=dtype)
    a_st = STensor.from_torch(adj, "A")
    g = torch.Generator().manual_seed(6)
    X = torch.randn(n, f_in, generator=g, dtype=torch.float64).to(dtype)
    labels = torch.randint(0, classes, (n,), generator=g)
    mask = torch.rand(n, generator=g) < 0.5
    mask[0] = True
    init = [
        (torch.randn(hidden, f_in, generator=g, dtype=torch.float64) * 0.3).to(dtype),
        (torch.randn(classes, hidden, generator=g, dtype=torch.float64) * 0.3).to(
            dtype
        ),
        torch.randn(hidden, generator=g, dtype=torch.float64).to(dtype) * 0.1,
        torch.randn(classes, generator=g, dtype=torch.float64).to(dtype) * 0.1,
    ]

    def step(prod):
        W1, W2, b1, b2 = (p.clone().requires_grad_(True) for p in init)
        x = F.linear(X, W1)
        x = prod(x) + b1
        x = F.relu(x)
        x = F.linear(x, W2)
        x = prod(x) + b2
        out = F.log_softmax(x, dim=1)
        loss = F.nll_loss(out[mask], labels[mask])
        loss.backward()
        return out.detach(), [p.grad for p in (W1, W2, b1, b2)]

    out, grads = step(lambda x: scorch.matmul(a_st, x, format="dd"))
    ref_out, ref_grads = step(lambda x: torch.sparse.mm(adj, x))
    tol = {
        torch.float32: dict(rtol=1e-4, atol=1e-5),
        torch.float64: dict(rtol=1e-10, atol=1e-12),
    }[dtype]
    torch.testing.assert_close(out, ref_out, **tol)
    for name, got, ref in zip(("W1", "W2", "b1", "b2"), grads, ref_grads):
        assert got is not None, name
        torch.testing.assert_close(
            got, ref, **tol, msg=lambda m, name=name: "%s: %s" % (name, m)
        )
