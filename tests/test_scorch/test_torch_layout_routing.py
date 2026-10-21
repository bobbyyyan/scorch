"""matmul and from_torch with torch's CSC, BSR and BSC layouts (patch 0073a).

matmul's first test (its tier 1) sends two torch tensors to torch.matmul unless one is COO
or CSR: torch sets `is_sparse` only for COO and `is_sparse_csr` only for CSR. So a pair
whose sparse operands are all CSC, BSR or BSC has always been torch's product, keywords
included, and it still is: what torch computes of those, scorch.matmul gives as torch gives
it, gradient included, and what torch refuses, torch still refuses. Sending CSC products to
scorch's own kernels waits for a measurement against torch's (patch 0073b).

What changes is from_torch: it adopts a 2-D CSC tensor with no copy, as the CSR arrays of
its transpose stored at mode order [1, 0]. So a CSC operand with a COO or CSR one, which
from_torch used to refuse by code, is computed by scorch, and so is every STensor operation
on an adopted CSC tensor (matmul, einsum, to_torch). Gradients through those products
behave as the same products with CSR in place of CSC do. BSR and BSC stay refused by
from_torch's code: they need blocked levels.

Two COO operands that are not both matrices are series step 19's (test_coo_operand_rank.py).

Values are compared with a float64 product of the same (rounded) operands, within the
forward error bound of a dot product of the inner length in the operands' dtype.
"""

import pytest
import torch

import scorch
import scorch_ops
from scorch.exceptions import ScorchError
from scorch.stensor import TORCH_ADOPTED_LAYOUTS

DTYPES = [torch.float32, torch.float64]
ROWS, INNER, COLS = 6, 4, 6  # block layouts need sides divisible by BLOCK
BLOCK = (2, 2)


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
    """A vector of both signs with no zero entry."""
    g = torch.Generator().manual_seed(seed)
    magnitude = torch.rand(n, generator=g, dtype=torch.float64) + 0.5
    sign = torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).to(torch.float64)
    return (magnitude * sign).to(dtype)


def _as_layout(dense, layout):
    if layout == "dense":
        return dense.clone()
    if layout == "coo":
        return dense.to_sparse().coalesce()
    if layout == "csr":
        return dense.to_sparse_csr()
    if layout == "csr32":
        csr = dense.to_sparse_csr()
        return torch.sparse_csr_tensor(
            csr.crow_indices().int(), csr.col_indices().int(), csr.values(), csr.shape
        )
    if layout == "csc":
        return dense.to_sparse_csc()
    if layout == "csc32":
        csc = dense.to_sparse_csc()
        return torch.sparse_csc_tensor(
            csc.ccol_indices().int(), csc.row_indices().int(), csc.values(), csc.shape
        )
    if layout == "bsr":
        return dense.to_sparse_bsr(BLOCK)
    if layout == "bsc":
        return dense.to_sparse_bsc(BLOCK)
    raise AssertionError(layout)


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


def _operands(left, right, dtype, seed):
    """(a, b, a64, b64): the operands in the named layouts and their float64 dense values.
    "vector" is a dense vector; "batch" a rank-3 dense tensor of two INNER x COLS
    matrices."""
    if left == "vector":
        a_d = _vector(ROWS, dtype, seed)
    else:
        a_d = _dense_matrix(ROWS, INNER, dtype, seed)
    if right == "vector":
        b_d = _vector(INNER, dtype, seed + 1)
    elif right == "batch":
        b_d = _dense_matrix(2 * INNER, COLS, dtype, seed + 1).reshape(2, INNER, COLS)
    else:
        b_d = _dense_matrix(INNER, COLS, dtype, seed + 1)
    if left == "vector":  # v @ M: M is ROWS x INNER, so v has ROWS entries
        b_d = _dense_matrix(ROWS, INNER, dtype, seed + 1)
        b = _as_layout(b_d, right)
        return a_d, b, a_d.to(torch.float64), b_d.to(torch.float64)
    a = _as_layout(a_d, left)
    b = b_d if right in ("vector", "batch") else _as_layout(b_d, right)
    return a, b, a_d.to(torch.float64), b_d.to(torch.float64)


def _outcome(fn):
    """("value", result) or ("raises", the exception's type and scorch code)."""
    try:
        return ("value", fn())
    except Exception as error:
        return ("raises", (type(error), getattr(error, "code", None)))


@pytest.fixture
def matmul_calls(monkeypatch):
    """(calls, real): every call of torch.matmul during the test, in order, and the real
    function."""
    calls = []
    real = torch.matmul

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "matmul", spy)
    return calls, real


@pytest.fixture
def torch_never_multiplies_csc(monkeypatch):
    """torch's product functions refuse, inside the test, any CSC argument: a product
    that should be scorch's and reaches torch is a routing defect."""

    def guard(real):
        def call(*args, **kwargs):
            for x in args:
                if isinstance(x, torch.Tensor) and x.layout is torch.sparse_csc:
                    raise AssertionError(f"{real.__name__} was handed a CSC operand")
            return real(*args, **kwargs)

        return call

    monkeypatch.setattr(torch, "matmul", guard(torch.matmul))
    monkeypatch.setattr(torch, "mm", guard(torch.mm))
    monkeypatch.setattr(torch.sparse, "mm", guard(torch.sparse.mm))


# --------------------------------------------------------------------------- #
# The layouts
# --------------------------------------------------------------------------- #


def test_the_adopted_layouts():
    assert TORCH_ADOPTED_LAYOUTS == (
        torch.strided,
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
    )


@pytest.mark.parametrize("layout", ["dense", "coo", "csr", "csc", "bsr", "bsc"])
def test_tier_one_lets_only_coo_and_csr_past(layout):
    """matmul's tier 1 reads `is_sparse or is_sparse_csr`: True for exactly COO and CSR,
    both of which from_torch adopts, so every torch operand tier 1 lets past is one
    from_torch takes."""
    x = _as_layout(_dense_matrix(ROWS, INNER, torch.float64, 0), layout)
    past_tier_one = x.is_sparse or x.is_sparse_csr
    assert past_tier_one == (layout in ("coo", "csr"))
    if past_tier_one:
        assert x.layout in TORCH_ADOPTED_LAYOUTS


@pytest.mark.parametrize("layout", ["bsr", "bsc"])
def test_from_torch_refuses_the_block_layouts_by_code(layout):
    x = _as_layout(_dense_matrix(ROWS, INNER, torch.float64, 1), layout)
    with pytest.raises(ScorchError) as error:
        scorch.from_torch(x)
    assert error.value.code == "torch_layout_unsupported"


# --------------------------------------------------------------------------- #
# CSC adopted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("layout", ["csc", "csc32"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_csc_is_adopted_as_the_transposes_csr_with_no_copy(layout, dtype):
    dense = _dense_matrix(ROWS, INNER, dtype, 2)
    a = _as_layout(dense, layout)
    s = scorch.from_torch(a)
    assert str(s.format) == "d,s"
    assert list(s.storage.index.mode_order) == [1, 0]
    assert tuple(s.shape) == (INNER, ROWS)  # storage order: columns, then rows
    ccol, row = scorch_ops.torch_sparse_index_members(a)
    stored = s.storage.index._mode_indices[1]  # as test_index_sharing_identity reads it
    assert stored[0].data_ptr() == ccol.data_ptr()
    assert stored[1].data_ptr() == row.data_ptr()
    assert s._raw_values.data_ptr() == a.values().data_ptr()
    torch.testing.assert_close(s.to_torch(in_place=False), dense, rtol=0, atol=0)


def test_csc_members_are_the_caller_arrays():
    """torch_sparse_index_members gives a CSC operand's own (ccol, row), as it gives a CSR
    operand's (crow, col): the arrays the caller built it from, sharing their counter.
    """
    dense = _dense_matrix(ROWS, INNER, torch.float64, 5)
    csc = dense.to_sparse_csc()
    ccol, row = csc.ccol_indices().clone(), csc.row_indices().clone()
    a = torch.sparse_csc_tensor(ccol, row, csc.values(), csc.shape)
    m_ccol, m_row = scorch_ops.torch_sparse_index_members(a)
    assert m_ccol.data_ptr() == ccol.data_ptr() and m_row.data_ptr() == row.data_ptr()
    before = m_row._version
    row.add_(0)
    assert m_row._version == before + 1


def test_csc_natural_order_and_explicit_orders():
    dense = _dense_matrix(ROWS, INNER, torch.float64, 3)
    a = dense.to_sparse_csc()
    natural = scorch.from_torch(a)
    explicit = scorch.from_torch(a, mode_order=[1, 0])
    assert list(explicit.storage.index.mode_order) == [1, 0]
    assert explicit._raw_values.data_ptr() == natural._raw_values.data_ptr()
    # An explicit identity order asks for row-major storage: a CSR copy.
    row_major = scorch.from_torch(a, mode_order=[0, 1])
    assert str(row_major.format) == "d,s"
    assert list(row_major.storage.index.mode_order or [0, 1]) == [0, 1]
    assert tuple(row_major.shape) == (ROWS, INNER)
    for s in (natural, explicit, row_major):
        torch.testing.assert_close(s.to_torch(in_place=False), dense, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["batched", "hybrid"])
def test_csc_of_another_rank_is_refused_by_code(kind):
    dense = _dense_matrix(ROWS, INNER, torch.float64, 4)
    if kind == "batched":
        a = torch.stack([dense, dense]).to_sparse_csc()
    else:
        a = dense.unsqueeze(-1).to_sparse_csc(dense_dim=1)
    with pytest.raises(ScorchError) as error:
        scorch.from_torch(a)
    assert error.value.code == "csc_rank_invalid"


# --------------------------------------------------------------------------- #
# Pairs with no COO or CSR operand: torch's product, as always
# --------------------------------------------------------------------------- #

TORCH_PAIRS = [
    (left, right)
    for left in ("dense", "vector", "csc", "csc32", "bsr", "bsc")
    for right in ("dense", "vector", "batch", "csc", "csc32", "bsr", "bsc")
    if {left, right} & {"csc", "csc32", "bsr", "bsc"}
    and not (left == "vector" and right in ("vector", "batch", "dense"))
]


@pytest.mark.parametrize("left,right", TORCH_PAIRS, ids=lambda p: str(p))
@pytest.mark.parametrize("dtype", DTYPES)
def test_pairs_with_no_coo_or_csr_operand_are_torchs(left, right, dtype, matmul_calls):
    """scorch.matmul hands torch.matmul the operands themselves, and returns what it
    returns or raises what it raises: whichever of these pairs this host's torch
    computes, scorch computes as torch does."""
    calls, real = matmul_calls
    a, b, _, _ = _operands(left, right, dtype, 30)
    want = _outcome(lambda: real(a, b))
    got = _outcome(lambda: scorch.matmul(a, b))
    assert len(calls) == 1 and calls[0][0] is a and calls[0][1] is b
    assert got[0] == want[0]
    if got[0] == "value":
        assert type(got[1]) is type(want[1]) and got[1].layout == want[1].layout
        torch.testing.assert_close(
            _to_dense(got[1]), _to_dense(want[1]), rtol=0, atol=0
        )
    else:
        assert got[1] == want[1]


KEYWORDS = [
    {"format": "dd"},
    {"format": "ds"},
    {"output_format": "ds"},
    {"output_mode_order": [1, 0]},
    {"format": "ds", "output_mode_order": [1, 0]},
    {"use_cache": False},
    {"time_dict": None},  # a fresh dict per call
]


def _with_fresh_dict(kwargs):
    if "time_dict" in kwargs:
        return dict(kwargs, time_dict={})
    return dict(kwargs)


@pytest.mark.parametrize("keywords", KEYWORDS, ids=lambda k: ",".join(k))
@pytest.mark.parametrize(
    "left,right",
    [
        ("csc", "dense"),
        ("dense", "csc"),
        ("bsr", "dense"),
        ("dense", "bsc"),
        ("bsc", "dense"),
        ("dense", "bsr"),
    ],
)
def test_keywords_on_those_pairs_do_what_they_do_for_dense_operands(
    left, right, keywords, matmul_calls
):
    """With a keyword these pairs take the route two dense tensors take (_dense_product):
    torch's product of the operands themselves, the keywords honoured on its result. So
    the result is what the same call gives for the operands' dense values, to rounding;
    and where torch does not multiply the pair, the call raises what torch raises.

    The one exception is that route's own: a sparse format with an output mode order is
    the generated kernel's, on the operands as from_torch makes them. That refused a CSC
    operand by code (torch_layout_unsupported), and now computes it; a block operand is
    refused by that code still."""
    calls, real = matmul_calls
    a, b, a64, b64 = _operands(left, right, torch.float64, 31)
    got = _outcome(lambda: scorch.matmul(a, b, **_with_fresh_dict(keywords)))
    generated = keywords.get("format") == "ds" and "output_mode_order" in keywords
    block = {left, right} & {"bsr", "bsc"}
    if generated and block:
        assert got[0] == "raises" and got[1][1] == "torch_layout_unsupported"
        return
    if not generated:
        torch_calls = list(calls)
        if _outcome(lambda: real(a, b))[0] == "raises":
            assert got[0] == "raises" and got[1] == _outcome(lambda: real(a, b))[1]
            return
        assert torch_calls and torch_calls[0][0] is a and torch_calls[0][1] is b
    want = _outcome(
        lambda: scorch.matmul(_to_dense(a), _to_dense(b), **_with_fresh_dict(keywords))
    )
    assert got[0] == want[0] == "value"
    out, ref = got[1], want[1]
    assert type(out) is type(ref)
    if isinstance(out, scorch.STensor):
        assert str(out.format) == str(ref.format)
        assert out.storage.index.mode_order == ref.storage.index.mode_order
    else:
        assert out.layout == ref.layout
        # An output mode order fixes the result's strides; otherwise they are torch's
        # product's own, which depend on the operands' layouts.
        if "output_mode_order" in keywords:
            assert out.stride() == ref.stride()
    _assert_within_bound(out, a64, b64, torch.float64)


@pytest.mark.parametrize(
    "left,right",
    [
        (left, right)
        for left in ("dense", "csc", "bsr", "bsc")
        for right in ("dense", "csc", "bsr", "bsc")
        if {left, right} != {"dense"}
    ],
)
def test_those_pairs_keep_torchs_gradient(left, right):
    """Both operands want a gradient. scorch.matmul's product and gradients are torch's,
    computed or refused as torch computes or refuses them."""

    def run(product):
        a, b, _, _ = _operands(left, right, torch.float64, 32)
        a.requires_grad_()
        b.requires_grad_()
        try:
            out = product(a, b)
        except Exception as error:
            return ("forward raises", type(error))
        upstream = out.detach().clone() if out.layout is not torch.strided else None
        if upstream is None:
            upstream = torch.ones_like(out)
        try:
            grads = torch.autograd.grad(out, (a, b), upstream)
        except Exception as error:
            return ("backward raises", type(error))
        return ("grads", tuple(_to_dense(g) for g in grads))

    got, want = run(scorch.matmul), run(torch.matmul)
    assert got[0] == want[0]
    if got[0] == "grads":
        for g, w in zip(got[1], want[1]):
            torch.testing.assert_close(g, w, rtol=0, atol=0)
    else:
        assert got[1] is want[1]


def test_dense_products_are_still_torchs(matmul_calls):
    calls, real = matmul_calls
    a = _dense_matrix(ROWS, INNER, torch.float64, 25)
    b = _dense_matrix(INNER, COLS, torch.float64, 26)
    torch.testing.assert_close(scorch.matmul(a, b), real(a, b), rtol=0, atol=0)
    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# CSC with a COO or CSR operand: scorch's, through from_torch's adoption
# --------------------------------------------------------------------------- #

MIXED_PAIRS = [
    ("csc", "csr"),
    ("csr", "csc"),
    ("csc", "coo"),
    ("coo", "csc"),
    ("csc32", "csr32"),
    ("csr32", "csc32"),
]


@pytest.mark.parametrize(
    "keywords", [{}, {"format": "dd"}, {"format": "ds"}], ids=lambda k: str(k)
)
@pytest.mark.parametrize("left,right", MIXED_PAIRS, ids=lambda p: str(p))
@pytest.mark.parametrize("dtype", DTYPES)
def test_csc_with_coo_or_csr_is_computed_by_scorch(
    left, right, dtype, keywords, torch_never_multiplies_csc
):
    """These were refused by from_torch's code (torch_layout_unsupported)."""
    a, b, a64, b64 = _operands(left, right, dtype, 10)
    for _ in range(3):  # cold, then the calls that find caches and plans
        out = scorch.matmul(a, b, **keywords)
        _assert_within_bound(out, a64, b64, dtype)


def _csr_in_place_of_csc(layout):
    return layout.replace("csc", "csr")


@pytest.mark.parametrize(
    "keywords", [{}, {"format": "dd"}, {"format": "ds"}], ids=lambda k: str(k)
)
@pytest.mark.parametrize("left,right", MIXED_PAIRS, ids=lambda p: str(p))
def test_a_product_with_csc_has_the_type_csr_gives(left, right, keywords):
    """The result's type, format and mode order are those of the same product with CSR
    in place of CSC."""
    a, b, _, _ = _operands(left, right, torch.float64, 11)
    ac, bc, _, _ = _operands(
        _csr_in_place_of_csc(left), _csr_in_place_of_csc(right), torch.float64, 11
    )
    out, ref = scorch.matmul(a, b, **keywords), scorch.matmul(ac, bc, **keywords)
    assert type(out) is type(ref)
    if isinstance(out, scorch.STensor):
        assert str(out.format) == str(ref.format)
        assert out.storage.index.mode_order == ref.storage.index.mode_order


def _layout_of_gradient(operand):
    """The layout of a torch sparse operand's gradient through scorch.matmul: torch's
    unmasked gradient (series step 21) in the operand's own layout, but for CSC, dense and
    strided. Torch's autograd engine takes no CSC-layout gradient for a CSC leaf
    (AccumulateGrad raises "Sparse CSC tensors do not have is_contiguous", uncoded), and
    torch's own sparse product with a CSC operand raises in its backward; dense is torch's
    layout for a CSC operand of a sparse times a dense product."""
    return torch.strided if operand.layout is torch.sparse_csc else operand.layout


def _assert_unmasked_gradients(a, b, a64, b64, upstream):
    """a.grad and b.grad, each in _layout_of_gradient's layout, equal to G @ B^T and A^T @ G
    for the upstream gradient G given, within the forward error bound in float64."""
    g = _to_dense(upstream).to(torch.float64)
    for operand, want_left, want_right in ((a, g, b64.T), (b, a64.T, g)):
        assert operand.grad is not None
        assert operand.grad.layout is _layout_of_gradient(operand)
        _assert_within_bound(operand.grad, want_left, want_right, torch.float64)


@pytest.mark.parametrize("keywords", [{}, {"format": "dd"}], ids=lambda k: str(k))
@pytest.mark.parametrize("left,right", MIXED_PAIRS, ids=lambda p: str(p))
def test_a_gradient_through_csc_with_coo_or_csr_is_as_csrs(left, right, keywords):
    """Both operands want a gradient. What the product gives -- an STensor, which carries
    no gradient to a torch operand, or a backward that runs -- is what the same product
    with CSR in place of CSC gives. A backward that runs gives each operand torch's
    unmasked gradient (_assert_unmasked_gradients) in .grad, where the engine accumulates
    it: a CSC operand's gradient in its own layout was an uncoded crash there."""

    def outcome(l, r):
        a, b, a64, b64 = _operands(l, r, torch.float64, 12)
        a.requires_grad_()
        b.requires_grad_()
        try:
            out = scorch.matmul(a, b, **keywords)
        except ScorchError as error:
            return ("forward code", error.code)
        _assert_within_bound(
            out.detach() if isinstance(out, torch.Tensor) else out,
            a64,
            b64,
            torch.float64,
        )
        if isinstance(out, scorch.STensor):
            return ("an STensor",)
        if not out.requires_grad:
            return ("a tensor without a gradient",)
        upstream = torch.ones_like(out)
        try:
            out.backward(upstream)
        except ScorchError as error:
            return ("backward code", error.code)
        _assert_unmasked_gradients(a, b, a64, b64, upstream)
        return ("backward ran",)

    assert outcome(left, right) == outcome(
        _csr_in_place_of_csc(left), _csr_in_place_of_csc(right)
    )


@pytest.mark.parametrize("result", ["ij,jk->ik", "ij,jk->k", "ij,jk->i"])
@pytest.mark.parametrize("csc_side", ["left", "right"])
def test_einsum_gives_a_csc_operand_a_dense_gradient(result, csc_side):
    """scorch.einsum over a torch CSC operand and a CSR one, both wanting a gradient, with
    a dense result: backward() runs and each .grad is torch's unmasked gradient, the CSC
    operand's dense (_layout_of_gradient). "ij,jk->k" and "ij,jk->i" sum over a label one
    operand alone names, whose gradient is computed dense and then laid out; for a CSC
    operand that was torch's to_sparse_csc, which the engine refuses."""
    left, right = ("csc", "csr") if csc_side == "left" else ("csr", "csc")
    a, b, a64, b64 = _operands(left, right, torch.float64, 14)
    a.requires_grad_()
    b.requires_grad_()
    out = scorch.einsum(result, a, b, format="d" * len(result.split("->")[1])).to_torch()
    upstream = torch.ones_like(out)
    out.backward(upstream)
    full = upstream
    if result.endswith("->k"):
        full = upstream.expand(a64.shape[0], -1)
    elif result.endswith("->i"):
        full = upstream.unsqueeze(1).expand(-1, b64.shape[1])
    full = full.to(torch.float64)
    for operand, want in ((a, full @ b64.T), (b, a64.T @ full)):
        assert operand.grad.layout is _layout_of_gradient(operand)
        torch.testing.assert_close(
            _to_dense(operand.grad), want, rtol=1e-12, atol=1e-12
        )


# test_schedule_force_routes.py's strip-mined SpMM: its generated C++ differs from every
# default kernel.
FORCED = scorch.Schedule(
    loop_order=("i", "j", "k"),
    tiles=(scorch.TileSpec("i", 4, placement="outermost", accum="direct"),),
    tag="tile-i-4",
)


@pytest.mark.parametrize("keywords", [{}, {"format": "dd"}], ids=lambda k: str(k))
@pytest.mark.parametrize(
    "left,right", MIXED_PAIRS + [("csc", "dense"), ("dense", "csc")], ids=lambda p: str(p)
)
def test_a_forced_schedule_treats_csc_as_csr(left, right, keywords):
    """Under schedule_force, a product with an adopted CSC operand is computed or refused
    as the same product with CSR in place of CSC is, with the same code: a sparse result
    refuses with tiling_sparse_result_unsupported, as for two CSR operands. A CSC pair
    with a dense operand is torch's product (matmul's tier 1) and keeps
    schedule_without_sparse_operand: a schedule names a sparse operand's loops."""

    def outcome(l, r):
        a, b, a64, b64 = _operands(l, r, torch.float64, 15)
        try:
            with scorch.schedule_force(FORCED):
                out = scorch.matmul(a, b, **keywords)
        except Exception as error:  # the compiler's refusals are not ScorchErrors
            code = getattr(error, "code", None)
            assert code is not None, f"an uncoded {type(error).__name__}: {error}"
            return ("code", code)
        _assert_within_bound(out, a64, b64, torch.float64)
        return ("computed", type(out), getattr(out, "layout", None))

    got = outcome(left, right)
    if "dense" in (left, right):
        assert got == ("code", "schedule_without_sparse_operand")
        return
    if not keywords:
        assert got == ("code", "tiling_sparse_result_unsupported")
    assert got == outcome(_csr_in_place_of_csc(left), _csr_in_place_of_csc(right))


NEW_ROWS = [2, 1, 0]


@pytest.mark.parametrize("route", ["operand_accessor", "caller_array", "values"])
@pytest.mark.parametrize(
    "index_dtype", [torch.int64, torch.int32], ids=["int64", "int32"]
)
@pytest.mark.parametrize("product", ["csc @ csr", "STensor @ dense"])
def test_a_write_to_a_csc_operand_reaches_the_next_product(route, index_dtype, product):
    """As test_index_writes does for CSR: warm the product, write in place, and the next
    product is computed from the matrix as it is now. "csc @ csr" adopts the torch CSC
    afresh on every call; "STensor @ dense" adopts it once and reuses the STensor, so
    whatever that STensor keeps (plans, cached copies) must see the write."""
    ccol = torch.tensor([0, 1, 2, 3], dtype=index_dtype)
    row = torch.tensor([0, 1, 2], dtype=index_dtype)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    a = torch.sparse_csc_tensor(ccol, row, values, size=(3, 3))
    b = torch.arange(12, dtype=torch.float64).reshape(3, 4).div(7.0)
    if product == "csc @ csr":
        left, right = a, b.to_sparse_csr()
    else:
        left, right = scorch.from_torch(a), b

    def check():
        # No keyword: a keyword skips the plan probe, and plans are what this checks.
        out = scorch.matmul(left, right)
        torch.testing.assert_close(_to_dense(out), a.to_dense() @ b, rtol=0, atol=1e-12)

    for _ in range(3):
        check()
    if route == "operand_accessor":
        a.row_indices().copy_(torch.tensor(NEW_ROWS, dtype=index_dtype))
    elif route == "caller_array":
        row.copy_(torch.tensor(NEW_ROWS, dtype=index_dtype))
    else:
        values.mul_(-2.0)
    check()


# --------------------------------------------------------------------------- #
# STensor operations on an adopted CSC tensor
# --------------------------------------------------------------------------- #


def _st(dense, layout):
    return scorch.from_torch(_as_layout(dense, layout))


def _stensor_cases(dtype):
    a = _dense_matrix(ROWS, INNER, dtype, 40)
    b = _dense_matrix(INNER, COLS, dtype, 41)
    x = _dense_matrix(COLS, ROWS, dtype, 42)
    v = _vector(INNER, dtype, 43)
    m = scorch.matmul
    e = scorch.einsum
    ij_jk = "ij,jk->ik"
    return {
        "cscST @ dense": (lambda: m(_st(a, "csc"), b), a, b),
        "cscST @ dense, dd": (lambda: m(_st(a, "csc"), b, format="dd"), a, b),
        "cscST @ dense, ds": (lambda: m(_st(a, "csc"), b, format="ds"), a, b),
        "dense @ cscST": (lambda: m(x, _st(a, "csc")), x, a),
        "dense @ cscST, dd": (lambda: m(x, _st(a, "csc"), format="dd"), x, a),
        "cscST @ vector": (lambda: m(_st(a, "csc"), v), a, v),
        "cscST @ csrST, ds": (
            lambda: m(_st(a, "csc"), _st(b, "csr"), format="ds"),
            a,
            b,
        ),
        "cscST @ csrST, dd": (
            lambda: m(_st(a, "csc"), _st(b, "csr"), format="dd"),
            a,
            b,
        ),
        "csrST @ cscST, ds": (
            lambda: m(_st(a, "csr"), _st(b, "csc"), format="ds"),
            a,
            b,
        ),
        "cscST @ cscST, ds": (
            lambda: m(_st(a, "csc"), _st(b, "csc"), format="ds"),
            a,
            b,
        ),
        "cscST @ csc": (lambda: m(_st(a, "csc"), _as_layout(b, "csc")), a, b),
        "csc @ cscST": (lambda: m(_as_layout(a, "csc"), _st(b, "csc")), a, b),
        "einsum cscST, dense, dd": (
            lambda: e(ij_jk, _st(a, "csc"), b, format="dd"),
            a,
            b,
        ),
        "einsum cscST, dense, ds": (
            lambda: e(ij_jk, _st(a, "csc"), b, format="ds"),
            a,
            b,
        ),
        "einsum dense, cscST, dd": (
            lambda: e(ij_jk, x, _st(a, "csc"), format="dd"),
            x,
            a,
        ),
        "einsum cscST, csrST, ds": (
            lambda: e(ij_jk, _st(a, "csc"), _st(b, "csr"), format="ds"),
            a,
            b,
        ),
        "einsum cscST, csrST, dd": (
            lambda: e(ij_jk, _st(a, "csc"), _st(b, "csr"), format="dd"),
            a,
            b,
        ),
        "einsum cscST, vector, d": (
            lambda: e("ij,j->i", _st(a, "csc"), v, format="d"),
            a,
            v,
        ),
        "einsum csc, dense, dd": (
            lambda: e(ij_jk, _as_layout(a, "csc"), b, format="dd"),
            a,
            b,
        ),
        "einsum csc, csr, ds": (
            lambda: e(ij_jk, _as_layout(a, "csc"), _as_layout(b, "csr"), format="ds"),
            a,
            b,
        ),
    }


STENSOR_CASES = list(_stensor_cases(torch.float64))


@pytest.mark.parametrize("case", STENSOR_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_stensor_operations_on_an_adopted_csc_compute(case, dtype):
    """Each of these was refused by from_torch's code (torch_layout_unsupported)."""
    fn, a, b = _stensor_cases(dtype)[case]
    for _ in range(2):  # cold, then warm
        _assert_within_bound(fn(), a.to(torch.float64), b.to(torch.float64), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_an_adopted_csc_transposes_and_adds(dtype):
    a = _dense_matrix(ROWS, INNER, dtype, 44)
    s = _st(a, "csc")
    torch.testing.assert_close(
        _to_dense(scorch.einsum("ij->ji", s, format="ds")), a.t(), rtol=0, atol=0
    )
    torch.testing.assert_close(_to_dense(s + _st(a, "csc")), a + a, rtol=0, atol=0)
    torch.testing.assert_close(_to_dense(s), a, rtol=0, atol=0)
