"""Batched sparse matmul: operands of rank three or more, by torch.matmul's rules.

``scorch.matmul`` computes a product with an operand of rank three or more as the einsum
of torch.matmul's rank rules (``matmul_rules.matmul_expression``): batch modes aligned at
the right and broadcast as torch broadcasts them, a batch mode of extent 1 against the
other operand's ``n`` given a label of its own that the product sums over.  Before, every
such product was refused with ``matmul_operand_rank_unsupported``.  torch itself refuses
these products for sparse operands (``expand is unsupported for Sparse tensors``), so the
reference is torch.matmul on the dense operands.

Values are compared with a float64 product of the same operands (rounded to the dtype
first), within the forward error bound of a dot product of the inner length in that dtype,
plus the rounding of the result.  The judge fails a NaN or an infinity by name before it
compares, and a test plants a NaN, an infinity and an error just above the bound to show
it fails each.  Gradients are checked by ``torch.autograd.gradcheck`` in float64.
"""

from __future__ import annotations

import itertools

import pytest
import torch

import scorch
from scorch import ops, stensor_grad
from scorch.exceptions import ScorchError
from scorch.matmul_rules import matmul_expression, operand_shape
from scorch.stensor import STensor

F64 = torch.float64
DTYPES = [torch.float32, torch.float64]
R, I, C = 4, 5, 3  # rows of the left operand, the contracted extent, columns of the right

# Each case: (left shape, right shape), torch.matmul's broadcasting in every form it has.
SHAPES = {
    "T@T": ((2, R, I), (2, I, C)),
    "T@M": ((2, R, I), (I, C)),
    "M@T": ((R, I), (2, I, C)),
    "T@v": ((2, R, I), (I,)),
    "v@T": ((I,), (2, I, C)),
    "4D@3D": ((3, 2, R, I), (2, I, C)),
    "1@n": ((1, R, I), (3, I, C)),
    "n@1": ((3, R, I), (1, I, C)),
    "1@1": ((1, R, I), (1, I, C)),
    "both broadcast": ((3, 1, R, I), (1, 2, I, C)),
}


# -- helpers ------------------------------------------------------------------------


def _dense(shape, dtype=F64, seed=0, density=0.5):
    """Entries of both signs, about half of them zero, and one empty row per matrix of
    two rows or more, rounded to ``dtype``."""
    g = torch.Generator().manual_seed(seed)
    values = torch.randn(shape, generator=g, dtype=F64)
    values = values * (torch.rand(shape, generator=g) < density)
    if len(shape) >= 2 and shape[-2] >= 2:
        values[..., 1, :] = 0.0
    return values.to(dtype)


def _format(family, rank):
    """A format of ``rank`` modes in ``family``: "csr" compresses the last mode only (CSR
    per batch), "dcsr" the last two, "sparse" every mode, "coo" one coordinate level per
    mode; a vector is a compressed "s" in every family but "coo"."""
    if rank == 1:
        return "o" if family == "coo" else "s"
    return {
        "csr": "d" * (rank - 1) + "s",
        "dcsr": "d" * (rank - 2) + "ss",
        "sparse": "s" * rank,
        "coo": "o" * rank,
    }[family]


def _stensor(dense, fmt, mode_order=None):
    with torch.no_grad():
        tensor = scorch.from_torch(dense.detach().clone())
        if mode_order is not None:
            tensor.change_mode_order(list(mode_order))
        if any(level != "d" for level in fmt):
            tensor.to_sparse(fmt)
    return tensor


def _as_dense(x):
    if type(x) is STensor:
        return x.to_torch(in_place=False)
    if x.layout is not torch.strided:
        return x.to_dense()
    return x


def _within_bound(out, a, b, dtype, what=""):
    """``out`` equals ``a @ b`` within ``gamma_n |a| @ |b| + u |a @ b|`` entrywise, where
    ``a`` and ``b`` are the operands as given (float64 copies of values already rounded to
    ``dtype``) and ``n`` the contracted extent: a dot product's forward error bound for any
    summation order, plus the rounding of the result to ``dtype``.  A NaN or an infinity
    fails by name, before the comparison, which is written as the condition that passes so
    that a NaN cannot satisfy it."""
    got = _as_dense(out).to(F64)
    a, b = a.to(F64), b.to(F64)
    ref = torch.matmul(a, b)
    assert got.shape == ref.shape, (what, got.shape, ref.shape)
    assert bool(torch.isfinite(got).all()), (what, "holds a NaN or an infinity")
    n = a.shape[-1]
    u = torch.finfo(dtype).eps / 2
    gamma = n * u / (1 - n * u)
    bound = gamma * torch.matmul(a.abs(), b.abs()) + u * ref.abs()
    assert bool(((got - ref).abs() <= 1.01 * bound).all()), (what, "exceeds the bound")


def _stored(result):
    """The stored coordinates of a sparse STensor result, as a set."""
    return set(_coordinates(result))


def _coordinates(result):
    """The stored coordinates of an STensor, in storage order."""
    keys = stensor_grad._Structure.of(result).keys()
    shape = tuple(result.logical_shape)
    out = []
    for key in keys.tolist():
        coordinate = []
        for extent in reversed(shape):
            coordinate.append(key % extent)
            key //= extent
        out.append(tuple(reversed(coordinate)))
    return out


def _outcome(fn):
    try:
        return ("value", fn())
    except ScorchError as error:
        return ("code", error.code)


# -- the rule itself ----------------------------------------------------------------


def _shape_pairs():
    """Every pair of ranks 1-4 with batch extents drawn from {1, 2, 3}, 0 included, and
    every alignment: the space the expression has to agree with torch.matmul on."""
    for p, q in itertools.product(range(1, 5), repeat=2):
        for a_batch in itertools.product((0, 1, 2, 3), repeat=max(p - 2, 0)):
            for b_batch in itertools.product((1, 2, 3), repeat=max(q - 2, 0)):
                a_shape = (*a_batch, 2, 3) if p >= 2 else (3,)
                b_shape = (*b_batch, 3, 4) if q >= 2 else (3,)
                yield a_shape, b_shape


def _broadcasts(a_batch, b_batch):
    """Whether two batch shapes broadcast: at every position aligned at the right, the
    extents are equal or one of them is 1."""
    for ea, eb in zip(reversed(a_batch), reversed(b_batch)):
        if ea != eb and 1 not in (ea, eb):
            return False
    return True


def test_the_expression_is_torch_matmul_for_every_rank_and_broadcast():
    pairs = list(_shape_pairs())
    refused = 0
    for a_shape, b_shape in pairs:
        a = torch.rand(a_shape, dtype=F64)
        b = torch.rand(b_shape, dtype=F64)
        expression = matmul_expression(a_shape, b_shape)
        assert expression is not None, (a_shape, b_shape)
        if not _broadcasts(a_shape[:-2], b_shape[:-2]):
            # torch refuses it, and einsum refuses the shared label's two extents.
            with pytest.raises(RuntimeError):
                torch.matmul(a, b)
            with pytest.raises(RuntimeError):
                torch.einsum(expression, a, b)
            refused += 1
            continue
        want = torch.matmul(a, b)
        got = torch.einsum(expression, a, b)
        assert got.shape == want.shape, (a_shape, b_shape, expression)
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    assert 0 < refused < len(pairs)


def test_the_expression_names_the_broadcast_mode_privately():
    # (1, R, I) against (3, I, C): the left batch mode is summed over (its own label),
    # the right one is the result's.
    assert matmul_expression((1, 4, 5), (3, 5, 6)) == "bij,ajk->aik"
    assert matmul_expression((3, 4, 5), (1, 5, 6)) == "aij,bjk->aik"
    assert matmul_expression((5,), (2, 5, 6)) == "j,ajk->ak"
    assert matmul_expression((2, 4, 5), (5,)) == "aij,j->ai"
    assert matmul_expression((2, 4, 5), (5, 6)) == "aij,jk->aik"
    assert matmul_expression((4, 5), (5, 6)) == "ij,jk->ik"


def test_a_rank_zero_operand_and_too_many_batch_modes_have_no_expression():
    assert matmul_expression((), (3,)) is None
    assert matmul_expression((3,), ()) is None
    # Each broadcast position uses two labels, and 49 are available.
    assert matmul_expression((1,) * 25 + (2, 3), (2,) * 25 + (3, 4)) is None
    assert matmul_expression((1,) * 24 + (2, 3), (2,) * 24 + (3, 4)) is not None


# -- values -------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", list(SHAPES))
@pytest.mark.parametrize("sparse", ["left", "right", "both"])
def test_batched_products_against_torch(dtype, case, sparse):
    a_shape, b_shape = SHAPES[case]
    a = _dense(a_shape, dtype, seed=1)
    b = _dense(b_shape, dtype, seed=2)
    sa = _stensor(a, _format("csr", a.dim())) if sparse in ("left", "both") else a
    sb = _stensor(b, _format("csr", b.dim())) if sparse in ("right", "both") else b
    out = scorch.matmul(sa, sb)
    _within_bound(out, a, b, dtype, case)


@pytest.mark.parametrize("family", ["csr", "dcsr", "sparse", "coo"])
@pytest.mark.parametrize("case", ["T@T", "1@n", "both broadcast"])
@pytest.mark.parametrize("sparse", ["left", "both"])
def test_every_format_family(family, case, sparse):
    a_shape, b_shape = SHAPES[case]
    a = _dense(a_shape, seed=3)
    b = _dense(b_shape, seed=4)
    sa = _stensor(a, _format(family, a.dim()))
    sb = _stensor(b, _format(family, b.dim())) if sparse == "both" else b
    _within_bound(scorch.matmul(sa, sb), a, b, F64, (family, case))


@pytest.mark.parametrize(
    "fmt,order",
    [("dds", (1, 0, 2)), ("dds", (2, 1, 0)), ("sss", (1, 2, 0)), ("dss", (0, 2, 1))],
)
@pytest.mark.parametrize("side", ["left", "right"])
def test_a_batch_mode_stored_inside(fmt, order, side):
    """The batch mode need not be the outermost stored level: the product is the same."""
    a = _dense((2, R, I), seed=5)
    b = _dense((2, I, C), seed=6)
    if side == "left":
        out = scorch.matmul(_stensor(a, fmt, order), b)
    else:
        out = scorch.matmul(a, _stensor(b, fmt, order))
    _within_bound(out, a, b, F64, (fmt, order, side))


def test_an_empty_batch_and_an_empty_matrix():
    for a_shape, b_shape in [((0, R, I), (1, I, C)), ((2, 0, I), (2, I, C))]:
        a = _dense(a_shape, seed=7)
        b = _dense(b_shape, seed=8)
        out = scorch.matmul(_stensor(a, _format("csr", 3)), b)
        assert tuple(_as_dense(out).shape) == tuple(torch.matmul(a, b).shape)


# -- result type --------------------------------------------------------------------


@pytest.mark.parametrize("case", ["T@T", "T@M", "M@T", "1@n"])
@pytest.mark.parametrize("family", ["csr", "coo"])
def test_the_result_is_the_einsum_it_is(case, family):
    """matmul's result has the type and format of the einsum it computes, as at rank two
    on the generated path: a dense format is a torch.Tensor, any other an STensor."""
    a_shape, b_shape = SHAPES[case]
    a = _stensor(_dense(a_shape, seed=9), _format(family, len(a_shape)))
    b = _stensor(_dense(b_shape, seed=10), _format(family, len(b_shape)))
    out = scorch.matmul(a, b)
    via = scorch.einsum(matmul_expression(a_shape, b_shape), a, b)
    if via.format.is_dense():
        assert type(out) is torch.Tensor and out.layout is torch.strided
    else:
        assert type(out) is STensor and str(out.format) == str(via.format)
        assert _stored(out) == _stored(via)


def test_the_stored_set_is_the_same_on_a_second_call():
    a = _stensor(_dense((2, R, I), seed=11), "dds")
    b = _stensor(_dense((2, I, C), seed=12), "dds")
    first, second = scorch.matmul(a, b), scorch.matmul(a, b)
    assert type(first) is STensor
    assert _coordinates(first) == _coordinates(second)
    torch.testing.assert_close(_as_dense(first), _as_dense(second), rtol=0, atol=0)


# -- torch operands -----------------------------------------------------------------


@pytest.mark.parametrize("case", ["T@T", "T@M", "M@T", "T@v", "v@T", "1@n"])
def test_rank3_torch_coo_operands_are_computed_as_the_same_stensors_are(case):
    a_shape, b_shape = SHAPES[case]
    a = _dense(a_shape, seed=13)
    b = _dense(b_shape, seed=14)
    got = scorch.matmul(a.to_sparse(), b.to_sparse())
    _within_bound(got, a, b, F64, case)
    same = scorch.matmul(scorch.from_torch(a.to_sparse()), scorch.from_torch(b.to_sparse()))
    assert type(got) is type(same)
    torch.testing.assert_close(_as_dense(got), _as_dense(same), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_torch_coo_batch_times_dense_batch(dtype):
    a = _dense((3, R, I), dtype, seed=15)
    b = _dense((3, I, C), dtype, seed=16)
    out = scorch.matmul(a.to_sparse(), b)
    assert type(out) is torch.Tensor and out.layout is torch.strided
    _within_bound(out, a, b, dtype)
    # torch computes this one product itself, as bmm, but not as matmul.
    torch.testing.assert_close(out, torch.bmm(a.to_sparse(), b))


# -- keywords -----------------------------------------------------------------------


def test_keywords_behave_as_at_rank_two():
    a_dense = _dense((2, R, I), seed=17)
    b = _dense((2, I, C), seed=18)
    a = _stensor(a_dense, "dds")
    want = torch.matmul(a_dense, b)
    out = scorch.matmul(a, b, format="ddd")
    assert type(out) is torch.Tensor
    torch.testing.assert_close(out, want)
    out = scorch.matmul(a, b, output_format="ddd")
    torch.testing.assert_close(_as_dense(out), want)
    sparse = scorch.matmul(a, _stensor(b, "dds"), format="dds")
    assert type(sparse) is STensor and str(sparse.format) == "d,d,s"
    torch.testing.assert_close(_as_dense(sparse), want)
    ordered = scorch.matmul(a, b, output_mode_order=[1, 0, 2])
    torch.testing.assert_close(_as_dense(ordered), want)
    times = {}
    scorch.matmul(a, b, time_dict=times)
    assert times["eval_time"] >= 0.0


def test_a_format_of_another_rank_is_refused_by_code():
    a = _stensor(_dense((2, R, I), seed=19), "dds")
    b = _dense((2, I, C), seed=20)
    assert _outcome(lambda: scorch.matmul(a, b, format="ds")) == (
        "code",
        "einsum_output_format_rank_mismatch",
    )


# -- refusals -----------------------------------------------------------------------


def test_batch_extents_that_do_not_broadcast_are_refused_by_code():
    a = _stensor(_dense((2, R, I), seed=21), "dds")
    b = _dense((3, I, C), seed=22)
    with pytest.raises(RuntimeError):
        torch.matmul(_as_dense(a), b)
    assert _outcome(lambda: scorch.matmul(a, b)) == (
        "code",
        "einsum_index_extent_mismatch",
    )


def test_contracted_extents_that_differ_are_refused_by_code():
    a = _stensor(_dense((2, R, I), seed=23), "dds")
    b = _dense((2, I + 1, C), seed=24)
    assert _outcome(lambda: scorch.matmul(a, b)) == (
        "code",
        "einsum_index_extent_mismatch",
    )


def test_a_rank_zero_operand_is_refused_by_code():
    a = _stensor(_dense((2, R, I), seed=25), "dds")
    assert _outcome(lambda: scorch.matmul(a, torch.tensor(2.0, dtype=F64))) == (
        "code",
        "matmul_operand_rank_unsupported",
    )


# -- gradients ----------------------------------------------------------------------


def _over(template, values):
    return STensor.from_components(
        tuple(template.shape),
        template.format,
        [list(arrays) for arrays in template._native_mode_indices()],
        values,
        mode_order=list(template.mode_order),
    )


def _start(template):
    return template._raw_values.clone().requires_grad_()


def _dense_of(result):
    if type(result) is STensor:
        return result.to_torch(in_place=False)
    return result


@pytest.mark.parametrize("case", ["T@T", "T@M", "T@v", "1@n", "n@1", "both broadcast"])
def test_gradient_of_a_sparse_left_operand_and_a_dense_right_one(case):
    a_shape, b_shape = SHAPES[case]
    template = _stensor(_dense(a_shape, seed=26), _format("csr", len(a_shape)))
    v = _start(template)
    b = _dense(b_shape, seed=27).requires_grad_()

    def scorch_f(v, b):
        return _dense_of(scorch.matmul(_over(template, v), b))

    assert torch.autograd.gradcheck(scorch_f, (v, b))


@pytest.mark.parametrize("case", ["T@T", "M@T", "v@T", "1@n", "n@1"])
def test_gradient_of_a_dense_left_operand_and_a_sparse_right_one(case):
    a_shape, b_shape = SHAPES[case]
    template = _stensor(_dense(b_shape, seed=28), _format("csr", len(b_shape)))
    v = _start(template)
    a = _dense(a_shape, seed=29).requires_grad_()

    def scorch_f(a, v):
        return _dense_of(scorch.matmul(a, _over(template, v)))

    assert torch.autograd.gradcheck(scorch_f, (a, v))


@pytest.mark.parametrize("family", ["csr", "sparse", "coo"])
def test_gradient_of_two_sparse_operands(family):
    ta = _stensor(_dense((1, R, I), seed=30), _format(family, 3))
    tb = _stensor(_dense((2, I, C), seed=31), _format(family, 3))
    va, vb = _start(ta), _start(tb)

    def scorch_f(va, vb):
        return _dense_of(scorch.matmul(_over(ta, va), _over(tb, vb)))

    assert torch.autograd.gradcheck(scorch_f, (va, vb))


def test_second_derivative_of_a_broadcast_product():
    template = _stensor(_dense((1, R, I), seed=32), "dds")
    v = _start(template)
    b = _dense((3, I, C), seed=33).requires_grad_()

    def scorch_f(v, b):
        return _dense_of(scorch.matmul(_over(template, v), b))

    assert torch.autograd.gradgradcheck(scorch_f, (v, b))


def test_gradient_through_torch_coo_operands():
    a = _dense((1, R, I), seed=34)
    b = _dense((2, I, C), seed=35).requires_grad_()
    sa = a.to_sparse().requires_grad_()
    out = scorch.matmul(sa, b)
    assert out.requires_grad
    w = torch.rand(out.shape, dtype=F64)
    (out * w).sum().backward()
    ad = a.clone().requires_grad_()
    bd = b.detach().clone().requires_grad_()
    (torch.matmul(ad, bd) * w).sum().backward()
    torch.testing.assert_close(b.grad, bd.grad)
    # A sparse operand's gradient is what torch.matmul gives a sparse operand: dense
    # and unmasked, every entry of the dense operand's gradient (only torch.sparse.mm
    # masks it to the stored coordinates), here summed over the broadcast mode.
    assert sa.grad.layout == torch.strided
    torch.testing.assert_close(sa.grad, ad.grad, rtol=1e-12, atol=1e-12)


# -- routes -------------------------------------------------------------------------
#
# A product whose storage allows it is computed by rank-two products
# (ops._batched_by_matrix_products); the rest by the batched einsum.  The spy records the
# operand shapes of every rank-two product a route makes (ops.matmul is the name the
# routes call; scorch.matmul, which the tests call, is bound at import and unchanged).

K = 6  # the free extent in the route cases, distinct from R, I and C


def _spy(monkeypatch):
    calls = []
    original = ops.matmul

    def recording(x, y, **kwargs):
        calls.append((tuple(operand_shape(x)), tuple(operand_shape(y))))
        return original(x, y, **kwargs)

    monkeypatch.setattr(ops, "matmul", recording)
    return calls


def _route_operand(kind, shape, seed):
    dense = _dense(shape, seed=seed)
    if kind == "dense":
        return dense, dense
    if kind == "torch coo":
        return dense.to_sparse(), dense
    return _stensor(dense, kind), dense


# (name, left kind, left shape, right kind, right shape, the rank-two products made)
ROUTES = [
    ("T@M", "dds", (3, R, I), "dense", (I, K), [((3 * R, I), (I, K))]),
    ("T@v", "dds", (3, R, I), "dense", (I,), [((3 * R, I), (I,))]),
    ("T@T", "dds", (3, R, I), "dense", (3, I, K), [((3 * R, 3 * I), (3 * I, K))]),
    ("4D@4D", "ddds", (2, 3, R, I), "dense", (2, 3, I, K), [((6 * R, 6 * I), (6 * I, K))]),
    ("n@1", "dds", (3, R, I), "dense", (1, I, K), [((3 * R, I), (I, K))]),
    ("M@T", "ds", (R, I), "dense", (3, I, K), [((R, I), (I, 3 * K))]),
    ("ss M@T", "ss", (R, I), "dense", (2, 3, I, K), [((R, I), (I, 6 * K))]),
    ("1@n", "dds", (1, R, I), "dense", (3, I, K), [((R, I), (I, 3 * K))]),
    ("coo T@T", "torch coo", (3, R, I), "dense", (3, I, K), [((3 * R, 3 * I), (3 * I, K))]),
    ("coo T@M", "torch coo", (3, R, I), "dense", (I, K), [((3 * R, I), (I, K))]),
    ("ooo T@T", "ooo", (3, R, I), "dense", (3, I, K), [((3 * R, 3 * I), (3 * I, K))]),
    # No rank-two route: the batched einsum.
    ("sss T@T", "sss", (3, R, I), "dense", (3, I, K), []),
    ("dss T@M", "dss", (3, R, I), "dense", (I, K), []),
    ("both broadcast", "ddds", (3, 1, R, I), "dense", (1, 2, I, K), []),
    ("T@T sparse right", "dense", (3, R, I), "dds", (3, I, K), []),
    ("T@T both sparse", "dds", (3, R, I), "dds", (3, I, K), []),
]


@pytest.mark.parametrize(
    "name,left,a_shape,right,b_shape,products", ROUTES, ids=[r[0] for r in ROUTES]
)
def test_each_route_where_the_storage_allows_it(
    monkeypatch, name, left, a_shape, right, b_shape, products
):
    a, a_dense = _route_operand(left, a_shape, 40)
    b, b_dense = _route_operand(right, b_shape, 41)
    calls = _spy(monkeypatch)
    out = scorch.matmul(a, b)
    assert calls == products, name
    _within_bound(out, a_dense, b_dense, F64, name)
    # The batched einsum (use_cache=False takes no other route) gives the same type and
    # values within the same bound.
    via_einsum = scorch.matmul(a, b, use_cache=False)
    assert type(via_einsum) is type(out), name
    _within_bound(via_einsum, a_dense, b_dense, F64, name)


@pytest.mark.parametrize(
    "name,left,a_shape,right,b_shape,products", ROUTES, ids=[r[0] for r in ROUTES]
)
def test_a_batched_product_is_not_a_view(name, left, a_shape, right, b_shape, products):
    """As torch.matmul's batched product is not.  A view made under ``no_grad``
    refuses an in-place write that records a gradient ("A view was created in no_grad
    mode and is being modified inplace with grad mode enabled"), which torch.matmul's
    product takes; a route's rank-two product reshaped is a view of it, so
    ``_generated_matmul`` returns it through ``ops._not_a_view``, as the batched
    einsum's dense result is returned."""
    a, a_dense = _route_operand(left, a_shape, 50)
    b, b_dense = _route_operand(right, b_shape, 51)
    with torch.no_grad():
        theirs = torch.matmul(a_dense, b_dense)
        out = scorch.matmul(a, b)
    assert theirs._base is None  # the property asserted below is torch's
    if products:
        assert type(out) is torch.Tensor, name
    if type(out) is not torch.Tensor:
        return  # a sparse result (two sparse operands) is an STensor, not a torch view
    assert out._base is None, f"{name}: the product is a view"
    w = torch.zeros(out.shape, dtype=out.dtype, requires_grad=True)
    out.add_(w)
    theirs.add_(w)
    assert out.requires_grad and theirs.requires_grad, name


def test_the_stacked_matrix_adopts_the_operands_arrays():
    a = _stensor(_dense((3, R, I), seed=42), "dds")
    stacked = ops._stacked_matrix(a)
    (pos, crd), (spos, scrd) = a._native_mode_indices()[-1], stacked._native_mode_indices()[-1]
    assert spos.data_ptr() == pos.data_ptr() and scrd.data_ptr() == crd.data_ptr()
    assert stacked._raw_values.data_ptr() == a._raw_values.data_ptr()
    assert tuple(stacked.logical_shape) == (3 * R, I)
    coo = scorch.from_torch(_dense((3, R, I), seed=43).to_sparse())
    from_coo = ops._stacked_matrix(coo)
    assert from_coo._native_mode_indices()[-1][1].data_ptr() == (
        coo._native_mode_indices()[-1][0].data_ptr()
    )
    torch.testing.assert_close(
        from_coo.to_torch(in_place=False).reshape(3, R, I), coo.to_torch(in_place=False)
    )


def test_a_stored_order_other_than_logical_takes_the_batched_einsum(monkeypatch):
    a = _stensor(_dense((3, R, I), seed=44), "dds", (1, 0, 2))
    assert ops._stacked_matrix(a) is None
    calls = _spy(monkeypatch)
    b = _dense((I, K), seed=45)
    _within_bound(scorch.matmul(a, b), _as_dense(a), b, F64)
    assert calls == []


def test_block_diagonal_indices_widen_where_int32_cannot_hold_them(monkeypatch):
    """Above 2**31 - 1 columns the offset column indices are int64; the bound is lowered
    here so a small product takes that branch."""
    monkeypatch.setattr(ops, "_INT32_MAX", 2 * I)
    a_dense = _dense((3, R, I), seed=46)
    b = _dense((3, I, K), seed=47)
    _within_bound(scorch.matmul(_stensor(a_dense, "dds"), b), a_dense, b, F64)


def test_an_empty_batch_takes_the_routes_too():
    for a_shape, b_shape in [((0, R, I), (I, K)), ((0, R, I), (0, I, K)), ((3, 0, I), (3, I, K))]:
        a_dense = _dense(a_shape, seed=48)
        b = _dense(b_shape, seed=49)
        out = scorch.matmul(_stensor(a_dense, _format("csr", 3)), b)
        assert tuple(out.shape) == tuple(torch.matmul(a_dense, b).shape)


# -- the judge's negative controls (mlsys's rule) ----------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_judge_fails_a_planted_nan_an_infinity_and_an_excess(dtype):
    a = _dense((2, R, I), dtype, seed=36)
    b = _dense((2, I, C), dtype, seed=37)
    ref = torch.matmul(a.to(F64), b.to(F64)).to(dtype)
    _within_bound(ref, a, b, dtype)
    u = torch.finfo(dtype).eps / 2
    n = I
    gamma = n * u / (1 - n * u)
    bound = gamma * torch.matmul(a.to(F64).abs(), b.to(F64).abs())
    bound = bound + u * torch.matmul(a.to(F64), b.to(F64)).abs()
    excess = torch.matmul(a.to(F64), b.to(F64)) + 1.5 * bound + 1e-300
    for bad in (float("nan"), float("inf")):
        planted = ref.clone()
        planted[0, 0, 0] = bad
        with pytest.raises(AssertionError):
            _within_bound(planted, a, b, dtype)
    with pytest.raises(AssertionError):
        _within_bound(excess.to(F64), a, b, dtype)
