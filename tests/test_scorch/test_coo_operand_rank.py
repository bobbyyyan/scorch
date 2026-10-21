"""matmul hands non-matrix COO operands to from_torch.

Two torch COO operands went over as CSR whatever their rank: matmul called to_sparse_csr on
both, and torch refuses that for any COO tensor whose modes are not exactly two sparse ones,
so a COO vector, a rank-3 COO tensor or a hybrid COO matrix ended in torch's uncoded
RuntimeError (the population scan's 180 uncoded keys: M@v, v@M, v@v, T@M, T@T). Now only two
COO matrices with both modes sparse go over as CSR; every other COO operand goes over as
from_torch makes it, and is computed or refused by code like the same STensor.

Values are compared with a float64 product of the same (rounded) operands, within the
forward error bound of a dot product of the inner length in the operands' dtype.
"""

import pytest
import torch

import scorch
from scorch.exceptions import ScorchError

DTYPES = [torch.float32, torch.float64]
ROWS, INNER, COLS = 6, 4, 5


def _dense_matrix(rows, cols, dtype, seed):
    """A matrix with about half its entries zero, one empty row and one empty column."""
    g = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, cols, generator=g, dtype=torch.float64)
    keep = torch.rand(rows, cols, generator=g) < 0.5
    values = values * keep
    values[1, :] = 0.0
    values[:, cols - 2] = 0.0
    return values.to(dtype)


def _vector(n, dtype, seed):
    """A vector of both signs with no zero entry, so every product term is stored."""
    g = torch.Generator().manual_seed(seed)
    magnitude = torch.rand(n, generator=g, dtype=torch.float64) + 0.5
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(torch.float64)
    return (magnitude * sign).to(dtype)


def _to_dense(x):
    if isinstance(x, scorch.STensor):
        return x.to_torch(in_place=False)
    if x.layout is not torch.strided:
        return x.to_dense()
    return x


def _assert_within_bound(out, a, b, dtype):
    """|out - a @ b| <= gamma_n |a| @ |b| + u |a @ b|, entrywise, a and b float64 copies of
    the operands as given (already rounded to dtype): the dot product's forward error
    bound for any summation order, plus the rounding of the result to dtype."""
    ref = a @ b
    n = a.shape[-1]
    u = torch.finfo(dtype).eps / 2
    gamma = n * u / (1 - n * u)
    bound = gamma * (a.abs() @ b.abs()) + u * ref.abs()
    got = _to_dense(out).to(torch.float64)
    assert got.shape == ref.shape
    err = (got - ref).abs()
    worst = (err - 1.01 * bound).max().item() if err.numel() else 0.0
    assert worst <= 0.0, f"error exceeds the forward error bound by {worst:.3e}"


@pytest.fixture
def csr_conversions(monkeypatch):
    """Every tensor to_sparse_csr is called on during the test, in call order."""
    seen = []
    original = torch.Tensor.to_sparse_csr

    def recording(self, *args, **kwargs):
        seen.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to_sparse_csr", recording)
    return seen


def _outcome(fn):
    try:
        return ("value", _to_dense(fn()))
    except ScorchError as error:
        return ("code", error.code)


@pytest.mark.parametrize("dtype", DTYPES)
def test_two_coo_matrices_still_go_over_as_csr(dtype, csr_conversions):
    a = _dense_matrix(ROWS, INNER, dtype, 20)
    b = _dense_matrix(INNER, COLS, dtype, 21)
    out = scorch.matmul(a.to_sparse(), b.to_sparse())
    _assert_within_bound(out, a.to(torch.float64), b.to(torch.float64), dtype)
    assert [t.layout for t in csr_conversions[:2]] == [torch.sparse_coo] * 2


@pytest.mark.parametrize("shapes", ["M@v", "v@M", "v@v"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_coo_vectors_are_computed(shapes, dtype, csr_conversions):
    m = _dense_matrix(ROWS, INNER, dtype, 30)
    v_inner = _vector(INNER, dtype, 31)
    v_rows = _vector(ROWS, dtype, 32)
    a, b = {"M@v": (m, v_inner), "v@M": (v_rows, m), "v@v": (v_inner, v_inner)}[shapes]
    out = scorch.matmul(a.to_sparse(), b.to_sparse())
    _assert_within_bound(out, a.to(torch.float64), b.to(torch.float64), dtype)
    assert all(t.dim() == 2 for t in csr_conversions if t.layout is torch.sparse_coo)


@pytest.mark.parametrize(
    "shapes,kwargs", [("M@v", {}), ("v@M", {"format": "d"}), ("v@v", {})]
)
def test_a_dense_coo_vector_product_stays_on_the_graph(shapes, kwargs):
    """A dense result of two COO operands that want a gradient carries a grad_fn, as a
    dense result of two COO matrices does. A sparse result (v@M with no keyword is a
    coordinate STensor) carries none, as a sparse result of two COO matrices carries
    none on this tree: that class belongs to the sparse x sparse gradient step."""
    m = _dense_matrix(ROWS, INNER, torch.float64, 40)
    v_inner = _vector(INNER, torch.float64, 41)
    v_rows = _vector(ROWS, torch.float64, 42)
    a, b = {"M@v": (m, v_inner), "v@M": (v_rows, m), "v@v": (v_inner, v_inner)}[shapes]
    sa, sb = a.to_sparse().requires_grad_(), b.to_sparse().requires_grad_()
    out = scorch.matmul(sa, sb, **kwargs)
    assert isinstance(out, torch.Tensor) and out.layout is torch.strided
    _assert_within_bound(out.detach(), a, b, torch.float64)
    assert out.requires_grad


@pytest.mark.parametrize("shapes", ["T@M", "M@T", "T@T", "T@v", "v@T"])
def test_rank3_coo_operands_are_refused_as_the_same_stensors_are(shapes):
    t = _dense_matrix(2 * ROWS, INNER, torch.float64, 50).reshape(2, ROWS, INNER)
    u = _dense_matrix(2 * INNER, COLS, torch.float64, 51).reshape(2, INNER, COLS)
    m_left = _dense_matrix(ROWS, INNER, torch.float64, 52)
    m_right = _dense_matrix(INNER, COLS, torch.float64, 53)
    v_inner = _vector(INNER, torch.float64, 54)
    a, b = {
        "T@M": (t, m_right),
        "M@T": (m_left, u),
        "T@T": (t, u),
        "T@v": (t, v_inner),
        "v@T": (v_inner, u),
    }[shapes]
    got = _outcome(lambda: scorch.matmul(a.to_sparse(), b.to_sparse()))
    want = _outcome(
        lambda: scorch.matmul(
            scorch.from_torch(a.to_sparse()), scorch.from_torch(b.to_sparse())
        )
    )
    assert got == ("code", "matmul_operand_rank_unsupported")
    assert want == got


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_hybrid_coo_matrix_is_refused_by_from_torchs_code(side, csr_conversions):
    hybrid = _dense_matrix(ROWS, INNER, torch.float64, 60).to_sparse(1)
    other = _dense_matrix(INNER, ROWS, torch.float64, 61).to_sparse()
    assert hybrid.dense_dim() == 1 and hybrid.dim() == 2
    a, b = (hybrid, other) if side == "left" else (other, hybrid)
    assert _outcome(lambda: scorch.matmul(a, b)) == ("code", "hybrid_coo_unsupported")
    assert not any(
        t.dense_dim() for t in csr_conversions if t.layout is torch.sparse_coo
    )
