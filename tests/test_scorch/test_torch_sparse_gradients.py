"""A torch sparse operand's gradient is torch's (series step 20).

Bobby, 2026-10-15: ``scorch.matmul`` matches torch, so a torch sparse operand of a
product gets the gradient ``torch.matmul`` gives it -- the one a dense operand would
get, not masked to its stored entries: for ``A @ B``, ``G @ B^T``, dense and strided for
a sparse times a dense product, and in the operand's own layout for a product of two
sparse operands of one layout. ``scorch.einsum`` follows the same rule (torch.einsum
raises on a sparse operand, and where torch raises scorch gives the unmasked gradient).
An STensor keeps the gradient of its stored values (test_stensor_autograd.py), and
``torch.sparse.mm``, which scorch does not define, keeps torch's own.

The products are dennis's gradient probe's (spdgrad_probe.py): every value positive and
the upstream gradient nonzero everywhere, given through ``torch.autograd.grad``, so the
unmasked gradient has no zero entry and differs from the masked one by O(1) off the
operand's pattern. float64 is held to 1e-12 and float32 to 1e-5, against float64
references from the float32 values.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import pytest
import torch

import scorch
import scorch_ops

pytestmark = pytest.mark.skipif(
    not hasattr(scorch_ops, "sddmm_csr"), reason="the gradient kernels are not built"
)

N, K, M, P = 40, 30, 20, 25


def _tolerance(dtype: torch.dtype) -> Dict[str, float]:
    bound = 1e-12 if dtype == torch.float64 else 1e-5
    return dict(rtol=bound, atol=bound)


def _positive(*shape: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)


def _pattern(rows: int = N, cols: int = K, seed: int = 20261014) -> torch.Tensor:
    """About 10% stored, and at least two entries in every row."""
    generator = torch.Generator().manual_seed(seed)
    stored = torch.rand(rows, cols, generator=generator) < 0.1
    for i in range(rows):
        stored[i, torch.randperm(cols, generator=generator)[:2]] = True
    return stored


PATTERN = _pattern()
VALUES = torch.where(
    PATTERN, _positive(N, K, seed=1), torch.zeros(N, K, dtype=torch.float64)
)


def _uncoalesced(dense: torch.Tensor) -> torch.Tensor:
    """Every third stored entry split into two at the same coordinate, shuffled."""
    coo = dense.to_sparse_coo().coalesce()
    indices, values = coo.indices(), coo.values()
    split = torch.arange(values.numel()) % 3 == 0
    rows = torch.cat([indices[:, ~split], indices[:, split], indices[:, split]], 1)
    parts = torch.cat([values[~split], 0.4 * values[split], 0.6 * values[split]])
    order = torch.randperm(parts.numel(), generator=torch.Generator().manual_seed(5))
    built = torch.sparse_coo_tensor(rows[:, order], parts[order], dense.shape)
    assert not built.is_coalesced()
    return built


LAYOUTS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "coo": lambda d: d.to_sparse_coo().coalesce(),
    "coo_unc": _uncoalesced,
    "csr64": lambda d: d.to_sparse_csr(),
    "csr32": lambda d: torch.sparse_csr_tensor(
        d.to_sparse_csr().crow_indices().int(),
        d.to_sparse_csr().col_indices().int(),
        d.to_sparse_csr().values(),
        d.shape,
    ),
    # Paired with a dense operand, a CSC operand stays torch's (scorch-merge's rule 1):
    # torch.matmul gets it, so its gradient is torch's by construction.
    "csc": lambda d: d.to_sparse_csc(),
}


@pytest.fixture(autouse=True)
def _shared_tensors_stay_constant():
    """VALUES and SECOND are every test's operands. ``.to(float64)`` and ``.double()``
    of a float64 tensor return the tensor itself, so a reference built from them without
    a clone and marked requires_grad marks the shared tensor, and every later test's
    operand then has history and is not a leaf (its .grad stays None)."""
    yield
    for name in ("VALUES", "SECOND"):
        shared = globals().get(name)
        assert shared is None or not shared.requires_grad, f"{name} requires grad"


# order: (the dense operand's shape, the upstream gradient's shape, the product)
ORDERS: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Callable[..., Any]]] = {
    "A@B": ((K, M), (N, M), lambda mm, a, x: mm(a, x)),
    "A@v": ((K,), (N,), lambda mm, a, x: mm(a, x)),
    "X@A": ((P, N), (P, K), lambda mm, a, x: mm(x, a)),
    "v@A": ((N,), (K,), lambda mm, a, x: mm(x, a)),
}


def _run(
    mm: Callable[..., Any], order: str, layout: str, dtype: torch.dtype, want: str
):
    """(product, A's gradient, the dense operand's gradient or None)."""
    x_shape, g_shape, product = ORDERS[order]
    a = LAYOUTS[layout](VALUES.to(dtype)).requires_grad_(True)
    x = _positive(*x_shape, seed=2).to(dtype).requires_grad_(want == "both")
    g = _positive(*g_shape, seed=3).to(dtype)
    out = product(mm, a, x)
    inputs = (a, x) if want == "both" else (a,)
    grads = torch.autograd.grad(out, inputs, g)
    return out.detach(), grads[0], grads[1] if want == "both" else None


@pytest.mark.parametrize("want", ["A", "both"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("order", list(ORDERS))
def test_matmul_gives_a_sparse_operand_torch_s_gradient(order, layout, dtype, want):
    got_out, got_a, got_x = _run(scorch.matmul, order, layout, dtype, want)
    want_out, want_a, want_x = _run(torch.matmul, order, layout, dtype, want)
    tol = _tolerance(dtype)
    torch.testing.assert_close(got_out.double(), want_out.double(), **tol)
    assert want_a.layout is torch.strided, "torch's gradient is dense (probe 84892)"
    assert got_a.layout is torch.strided and got_a.shape == (N, K)
    torch.testing.assert_close(got_a.double(), want_a.double(), **tol)
    # Not masked: nonzero off A's pattern, as every upstream value is.
    assert bool((got_a[~PATTERN] != 0).all())
    if want == "both":
        torch.testing.assert_close(got_x.double(), want_x.double(), **tol)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("layout", ["coo", "coo_unc", "csr64", "csr32"])
@pytest.mark.parametrize("order", ["A@B", "X@A"])
def test_the_sparse_side_is_torch_s_own_call(order, layout, dtype):
    """A's gradient is computed by the torch.mm call torch's MmBackward0 makes
    (``G @ B^T``, ``X^T @ G``), on the same tensors, so it is torch's bit for bit,
    strides included."""
    _, got, _ = _run(scorch.matmul, order, layout, dtype, "both")
    _, want, _ = _run(torch.matmul, order, layout, dtype, "both")
    assert got.stride() == want.stride()
    assert torch.equal(got, want)


def test_entry_points_scorch_does_not_define_are_torch_s():
    """``import scorch as torch`` reaches torch's own objects for these, so their
    gradients are torch's by construction: torch.sparse.mm and addmm keep the
    stored-entries gradient, torch.mm and F.linear the dense one."""
    for name in ("mm", "addmm", "mv", "addmv"):
        assert getattr(scorch, name) is getattr(torch, name)
    assert scorch.sparse.mm is torch.sparse.mm
    assert scorch.sparse.addmm is torch.sparse.addmm
    assert scorch.nn.functional.linear is torch.nn.functional.linear


# --------------------------------------------------------------------------- #
# Products the einsum's derivative records (keywords, the generated kernel)
# --------------------------------------------------------------------------- #

# format="dd" on a sparse x dense product is the plain call (patch 0072), so it is
# recorded by autograd.py's Function; the others by the einsum's derivative.
KEYWORDS: Dict[str, Dict[str, Any]] = {
    "format": {"format": "dd"},
    "use_cache": {"use_cache": False},
    "output_mode_order": {"output_mode_order": [1, 0]},
    "sparse result": {"format": "ds"},
}


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("layout", ["coo", "csr64"])
@pytest.mark.parametrize("keyword", list(KEYWORDS))
def test_a_keyword_product_gives_torch_s_gradient(keyword, layout, dtype):
    a = LAYOUTS[layout](VALUES.to(dtype)).requires_grad_(True)
    x = _positive(K, M, seed=2).to(dtype).requires_grad_(True)
    g = _positive(N, M, seed=3).to(dtype)
    out = scorch.matmul(a, x, **KEYWORDS[keyword])
    dense = out.to_torch() if isinstance(out, scorch.STensor) else out
    ga, gx = torch.autograd.grad(dense, (a, x), g)
    a64, x64, g64 = VALUES.to(dtype).double(), x.detach().double(), g.double()
    tol = _tolerance(dtype)
    assert ga.layout is torch.strided
    torch.testing.assert_close(ga.double(), g64 @ x64.t(), **tol)
    torch.testing.assert_close(gx.double(), a64.t() @ g64, **tol)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_a_keyword_product_differentiates_twice(dtype):
    """With create_graph, x's gradient ``A^T @ G`` is computed with A itself, so its
    derivative with respect to A is torch's too, against dense autograd."""
    a = LAYOUTS["csr64"](VALUES.to(dtype)).requires_grad_(True)
    x = _positive(K, M, seed=2).to(dtype).requires_grad_(True)
    (gx,) = torch.autograd.grad(
        scorch.matmul(a, x, use_cache=False).pow(2).sum(), (x,), create_graph=True
    )
    (second,) = torch.autograd.grad(gx.sum(), (a,))
    d = VALUES.to(dtype).clone().requires_grad_(True)
    xr = x.detach().clone().requires_grad_(True)
    (gx_ref,) = torch.autograd.grad((d @ xr).pow(2).sum(), (xr,), create_graph=True)
    (second_ref,) = torch.autograd.grad(gx_ref.sum(), (d,))
    assert second.layout is torch.strided
    torch.testing.assert_close(second, second_ref, **_tolerance(dtype))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("layout", ["coo", "csr64", "csr32"])
@pytest.mark.parametrize("entry", ["matmul", "use_cache=False"])
def test_a_gradient_penalty_through_the_sparse_operand_reaches_the_dense_one(
    entry, layout, dtype
):
    """A gradient penalty (0082b's route R9): ``gw, ga = grad(loss, (w, a),
    create_graph=True)``, then ``loss + (|gw|^2 + |ga|^2) / 2`` is differentiated. ``ga``
    depends on w, so w's gradient is ``(A + G_A)^T R`` with ``G_A = R w^T``, as dense
    autograd gives. Before step 20 A's gradient was its stored values' and had no graph,
    so the penalty's term never reached w: w.grad was ``A^T R``, with no error."""
    a = LAYOUTS[layout](VALUES.to(dtype)).requires_grad_(True)
    w = _positive(K, M, seed=2).to(dtype).requires_grad_(True)
    r = _positive(N, M, seed=3).to(dtype)
    kwargs = {} if entry == "matmul" else {"use_cache": False}
    loss = (scorch.matmul(a, w, **kwargs) * r).sum()
    gw, ga = torch.autograd.grad(loss, (w, a), create_graph=True)
    assert ga.layout is torch.strided and ga.grad_fn is not None
    (loss + 0.5 * ((gw * gw).sum() + (ga * ga).sum())).backward()
    d = VALUES.to(dtype).double().clone().requires_grad_(True)
    w64 = w.detach().double().requires_grad_(True)
    loss64 = ((d @ w64) * r.double()).sum()
    gw64, gd64 = torch.autograd.grad(loss64, (w64, d), create_graph=True)
    (loss64 + 0.5 * ((gw64 * gw64).sum() + (gd64 * gd64).sum())).backward()
    tol = _tolerance(dtype)
    torch.testing.assert_close(w.grad.double(), w64.grad, **tol)
    torch.testing.assert_close(a.grad.double(), d.grad, **tol)


def test_a_values_write_before_the_backward_is_refused_as_torch_refuses_it():
    a = LAYOUTS["csr64"](VALUES.clone()).requires_grad_(True)
    x = _positive(K, M, seed=2).requires_grad_(True)
    out = scorch.matmul(a, x, use_cache=False)
    with torch.no_grad():
        a.values().mul_(2.0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


# --------------------------------------------------------------------------- #
# scorch.einsum, and products of two sparse operands
# --------------------------------------------------------------------------- #

SECOND = torch.where(
    _pattern(K, M, seed=7),
    _positive(K, M, seed=8),
    torch.zeros(K, M, dtype=torch.float64),
)


def _implied(a_dense: torch.Tensor, b_dense: torch.Tensor) -> torch.Tensor:
    """The coordinates of A @ B the operands' structures imply, whatever their
    values."""
    return ((a_dense != 0).double() @ (b_dense != 0).double()) != 0


def _dense_of(out: Any) -> torch.Tensor:
    """A product as a dense tensor with its history: an STensor's to_torch, a torch
    sparse tensor's to_dense (whose backward keeps the result's stored coordinates, as
    the gradient of an STensor's stored values does), a dense tensor itself."""
    if isinstance(out, scorch.STensor):
        return out.to_torch()
    return out.to_dense() if out.layout is not torch.strided else out


def _torch_matmul_gradients(a: torch.Tensor, b: torch.Tensor, g: torch.Tensor):
    """torch.matmul's gradients of ``sum(dense(a @ b) * g)`` for both operands, or None
    where torch cannot compute the product on this host (a mixed sparse pair; CSR x CSR
    on a torch built without MKL, as the ARM host's is)."""
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)
    try:
        out = torch.matmul(a, b)
    except (RuntimeError, NotImplementedError):
        return None
    return torch.autograd.grad((_dense_of(out) * g).sum(), (a, b))


# einsum products with a torch.matmul form: (expression, the dense operand's shape, the
# upstream gradient's shape, the sparse operand's side). Each is called with a dense
# result (format "d..."), as torch.matmul's is: a sparse result stores only the
# coordinates the structures imply (X @ A has no column where A has none), so its
# gradient is G on those alone, and it would be a different program from torch's.
MATMUL_EINSUMS = {
    "A@B": ("ij,jk->ik", (K, M), (N, M), "left"),
    "A@v": ("ij,j->i", (K,), (N,), "left"),
    "X@A": ("pi,ij->pj", (P, N), (P, K), "right"),
    "v@A": ("i,ij->j", (N,), (K,), "right"),
}


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("layout", ["coo", "coo_unc", "csr64", "csr32"])
@pytest.mark.parametrize("product", list(MATMUL_EINSUMS))
def test_einsum_with_a_matmul_form_gives_torch_matmul_s_gradient(
    product, layout, dtype
):
    expression, shape, g_shape, side = MATMUL_EINSUMS[product]
    a = LAYOUTS[layout](VALUES.to(dtype)).requires_grad_(True)
    x = _positive(*shape, seed=2).to(dtype).requires_grad_(True)
    g = _positive(*g_shape, seed=3).to(dtype)
    operands = (a, x) if side == "left" else (x, a)
    out = scorch.einsum(expression, *operands, format="d" * len(g_shape))
    got = torch.autograd.grad((_dense_of(out) * g).sum(), operands)
    want = _torch_matmul_gradients(*operands, g)
    assert want is not None, "torch.matmul computes every sparse x dense product here"
    tol = _tolerance(dtype)
    for mine, torch_s in zip(got, want):
        assert mine.layout is torch_s.layout is torch.strided
        torch.testing.assert_close(mine.double(), torch_s.double(), **tol)


# einsum products with no torch.matmul form, with a dense result (format="d..."), so the
# upstream gradient is wanted everywhere: (expression, the dense operands' shapes,
# the result's shape).
OTHER_EINSUMS = {
    "row sums": ("ij->i", (), (N,)),
    "elementwise": ("ij,ij->ij", ((N, K),), (N, K)),
    "chain": ("ij,jk,k->i", ((K, M), (M,)), (N,)),
    "outer": ("ij,k->ijk", ((M,),), (N, K, M)),
}


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("layout", ["coo", "csr64"])
@pytest.mark.parametrize("product", list(OTHER_EINSUMS))
def test_einsum_without_a_matmul_form_gives_the_dense_gradient(product, layout, dtype):
    """Against dense float64 autograd of the same einsum: unmasked and strided, as the
    gradient of a sparse operand with every other operand dense."""
    expression, shapes, result_shape = OTHER_EINSUMS[product]
    a = LAYOUTS[layout](VALUES.to(dtype)).requires_grad_(True)
    others = [_positive(*s, seed=2 + i).to(dtype) for i, s in enumerate(shapes)]
    g = _positive(*result_shape, seed=3).to(dtype)
    out = scorch.einsum(expression, a, *others, format="d" * len(result_shape))
    (got,) = torch.autograd.grad((_dense_of(out) * g).sum(), (a,))
    d = VALUES.to(dtype).double().clone().requires_grad_(True)
    ref = torch.einsum(expression, d, *(o.double() for o in others))
    (want,) = torch.autograd.grad((ref * g.double()).sum(), (d,))
    assert got.layout is torch.strided and got.shape == (N, K)
    torch.testing.assert_close(got.double(), want, **_tolerance(dtype))


# (A's layout, B's layout): the same layout gives torch's layout for each gradient; a
# mixed pair, which torch.matmul cannot compute, gives each operand's own.
SPARSE_PAIRS = [("csr64", "csr64"), ("coo", "coo"), ("csr64", "coo"), ("coo", "csr64")]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("entry", ["matmul", "einsum"])
@pytest.mark.parametrize("pair", SPARSE_PAIRS, ids=lambda p: "x".join(p))
def test_a_product_of_two_sparse_operands_gives_each_its_layout(pair, entry, dtype):
    """Each operand's gradient is unmasked and in its own layout, storing the
    coordinates the factors' structures imply. The result -- an STensor, or for matmul
    of two torch CSR operands torch's CSR tensor -- has stored values that carry the
    graph, so its gradient is G on its own stored coordinates, as
    torch's to_dense gives a sparse result's (masked_grad). Where torch computes the
    product on this host its gradients are the reference (layout, values, stored
    count); elsewhere -- torch raises -- float64."""
    a_dense, b_dense = VALUES.to(dtype), SECOND.to(dtype)
    a = LAYOUTS[pair[0]](a_dense).requires_grad_(True)
    b = LAYOUTS[pair[1]](b_dense).requires_grad_(True)
    out = scorch.matmul(a, b) if entry == "matmul" else scorch.einsum("ij,jk->ik", a, b)
    assert out.requires_grad
    if entry == "matmul" and pair == ("csr64", "csr64"):
        # Two torch CSR operands' product is torch's CSR tensor (patch A).
        assert type(out) is torch.Tensor and out.layout is torch.sparse_csr
        dense = out.to_dense()
    else:
        assert isinstance(out, scorch.STensor)
        dense = out.to_torch()
    g = _positive(N, M, seed=3).to(dtype)
    ga, gb = torch.autograd.grad((dense * g).sum(), (a, b))
    g64 = g.double() * _implied(a_dense, b_dense)
    a64, b64 = a_dense.double(), b_dense.double()
    tol = _tolerance(dtype)
    for got, operand, want, implied in (
        (ga, a, g64 @ b64.t(), _implied(g64, b64.t())),
        (gb, b, a64.t() @ g64, _implied(a64.t(), g64)),
    ):
        assert got.layout is operand.layout
        torch.testing.assert_close(got.to_dense().double(), want, **tol)
        assert got._nnz() == int(implied.sum())
    torch_s = _torch_matmul_gradients(a, b, g)
    if torch_s is not None:
        for got, want in zip((ga, gb), torch_s):
            assert got.layout is want.layout
            torch.testing.assert_close(
                got.to_dense().double(), want.to_dense().double(), **tol
            )
            count = want.coalesce()._nnz() if want.is_sparse else want._nnz()
            assert got._nnz() == count


@pytest.mark.parametrize("layout", ["coo", "csr64"])
def test_an_stensor_operand_keeps_its_stored_entries_gradient(layout):
    """Beside a torch sparse operand, an STensor is masked by its own contract."""
    a = LAYOUTS[layout](VALUES.clone()).requires_grad_(True)
    s = scorch.STensor.from_torch(SECOND.to_sparse_csr()).requires_grad_(True)
    out = scorch.matmul(a, s)
    g = _positive(N, M, seed=3)
    (out.to_torch() * g).sum().backward()
    g64 = g * _implied(VALUES, SECOND)
    assert a.grad.layout is a.layout
    torch.testing.assert_close(
        a.grad.to_dense(), g64 @ SECOND.t(), rtol=1e-12, atol=1e-12
    )
    want_s = (VALUES.t() @ g64) * (SECOND != 0)
    torch.testing.assert_close(s.grad.to_torch(), want_s, rtol=1e-12, atol=1e-12)


@pytest.mark.perf
def test_a_large_operand_s_gradient_is_torch_s():
    """2^20 rows: A's gradient is dense rows x cols (as torch's is), the dense side's
    the transposed SpMM; both against torch.matmul's."""
    rows, cols, width = 1 << 20, 16, 64
    generator = torch.Generator().manual_seed(11)
    nnz_per_row = 4
    crow = torch.arange(0, (rows + 1) * nnz_per_row, nnz_per_row)
    col = torch.randint(0, cols, (rows * nnz_per_row,), generator=generator)
    col = col.view(rows, nnz_per_row).sort(1).values.reshape(-1)
    values = torch.rand(rows * nnz_per_row, generator=generator)
    a = torch.sparse_csr_tensor(crow, col, values, (rows, cols))
    x = torch.rand(cols, width, generator=generator)
    g = torch.rand(rows, width, generator=generator)

    def run(mm):
        aa = a.detach().requires_grad_(True)
        xx = x.detach().requires_grad_(True)
        return torch.autograd.grad(mm(aa, xx), (aa, xx), g)

    (ga, gx), (wa, wx) = run(scorch.matmul), run(torch.matmul)
    assert ga.layout is torch.strided and torch.equal(ga, wa)
    torch.testing.assert_close(gx, wx, rtol=1e-4, atol=1e-4)
