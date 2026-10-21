"""Gradients through scorch.matmul's sparse x dense products (scorch/autograd.py).

Every product is checked against torch's own: the forward value, the dense operand's
gradient and, for a torch sparse operand that requires grad, A.grad (a sparse tensor with
A's structure, as torch.sparse.mm gives). gradcheck and gradgradcheck cover the
derivatives numerically, first and second order, and forward-mode AD.
"""

import contextlib
import gc
import weakref
import pickle

import pytest
import torch
import torch.autograd.forward_ad as fwAD

import scorch
import scorch_ops
from scorch import STensor
from scorch.exceptions import TensorIndexError
from scorch import autograd as grad_mod
from scorch import einsum_grad as einsum_grad_mod
from scorch import plan as plan_mod
from scorch.plan import plans_of

pytestmark = pytest.mark.skipif(
    not hasattr(scorch_ops, "sddmm_csr"), reason="the gradient kernels are not built"
)


def dense_matrix(rows=7, cols=5, density=0.5, dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((rows, cols), generator=generator) < density
    return (torch.rand((rows, cols), generator=generator) * mask).to(dtype)


def rand(*shape, dtype=torch.float64, seed=1):
    return torch.rand(shape, generator=torch.Generator().manual_seed(seed)).to(dtype)


def tolerance(dtype):
    return (
        dict(atol=1e-5, rtol=1e-5)
        if dtype == torch.float32
        else dict(atol=1e-10, rtol=1e-10)
    )


# How each kind of sparse operand is built from a dense matrix, and whether a gradient can
# be asked of the operand itself (an STensor stores its values detached).
OPERANDS = {
    "torch_csr": (lambda d: d.to_sparse_csr(), True),
    "torch_csr_int32": (
        lambda d: torch.sparse_csr_tensor(
            d.to_sparse_csr().crow_indices().int(),
            d.to_sparse_csr().col_indices().int(),
            d.to_sparse_csr().values(),
            d.shape,
        ),
        True,
    ),
    "torch_coo": (lambda d: d.to_sparse_coo(), True),
    "torch_coo_uncoalesced": (
        lambda d: torch.sparse_coo_tensor(
            d.to_sparse_coo().indices().flip(1),
            d.to_sparse_coo().values().flip(0),
            d.shape,
        ),
        True,
    ),
    "stensor_csr": (lambda d: STensor.from_torch(d.to_sparse_csr()), False),
    "stensor_csc": (
        lambda d: STensor.from_torch(d, mode_order=[1, 0]).to_sparse("ds"),
        False,
    ),
    "stensor_coo": (lambda d: STensor.from_torch(d.to_sparse_coo()), False),
    "stensor_dcsr": (lambda d: STensor.from_torch(d).to_sparse("ss"), False),
}


def check_against_torch(kind, dtype, side, dense_shape, sparse_grad):
    build, can_grad = OPERANDS[kind]
    if sparse_grad and not can_grad:
        pytest.skip("an STensor carries no gradient of its own")
    d = dense_matrix(dtype=dtype)
    a = build(d)
    if sparse_grad:
        a.requires_grad_()
    w = rand(*dense_shape, dtype=dtype).requires_grad_()
    out = scorch.matmul(a, w) if side == "left" else scorch.matmul(w, a)

    d_ref = d.clone().requires_grad_(sparse_grad)
    w_ref = w.detach().clone().requires_grad_()
    ref = d_ref @ w_ref if side == "left" else w_ref @ d_ref
    g = rand(*ref.shape, dtype=dtype, seed=2)
    (out * g).sum().backward()
    (ref * g).sum().backward()

    tol = tolerance(dtype)
    assert type(out) is torch.Tensor and out.grad_fn is not None
    torch.testing.assert_close(out, ref, **tol)
    torch.testing.assert_close(w.grad, w_ref.grad, **tol)
    if sparse_grad:
        assert a.grad.layout == a.layout
        torch.testing.assert_close(a.grad.to_dense(), d_ref.grad * (d != 0), **tol)
        if a.layout is torch.sparse_csr:
            assert torch.equal(a.grad.crow_indices(), a.crow_indices())
            assert torch.equal(a.grad.col_indices(), a.col_indices())


@pytest.mark.parametrize("kind", sorted(OPERANDS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "side,dense_shape",
    [("left", (5, 3)), ("left", (5,)), ("right", (4, 7)), ("right", (7,))],
)
def test_dense_gradient_matches_torch(kind, dtype, side, dense_shape):
    check_against_torch(kind, dtype, side, dense_shape, sparse_grad=False)


@pytest.mark.parametrize("kind", [k for k, (_, can) in sorted(OPERANDS.items()) if can])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "side,dense_shape",
    [("left", (5, 3)), ("left", (5,)), ("right", (4, 7)), ("right", (7,))],
)
def test_sparse_gradient_matches_torch(kind, dtype, side, dense_shape):
    check_against_torch(kind, dtype, side, dense_shape, sparse_grad=True)


# --------------------------------------------------------------------------- #
# Numerical checks: first and second derivatives, forward mode
# --------------------------------------------------------------------------- #


def _csr_parts():
    a = dense_matrix(6, 5).to_sparse_csr()
    return a.crow_indices(), a.col_indices(), a.values().clone()


def test_gradcheck_and_gradgradcheck():
    crow, col, values = _csr_parts()

    def left(v, w):
        return scorch.matmul(torch.sparse_csr_tensor(crow, col, v, (6, 5)), w)

    def right(v, x):
        return scorch.matmul(x, torch.sparse_csr_tensor(crow, col, v, (6, 5)))

    v = values.requires_grad_()
    w = rand(5, 3).requires_grad_()
    x = rand(4, 6).requires_grad_()
    assert torch.autograd.gradcheck(left, (v, w))
    assert torch.autograd.gradcheck(right, (v, x))
    assert torch.autograd.gradgradcheck(left, (v, w))
    assert torch.autograd.gradgradcheck(right, (v, x))


def test_forward_mode_ad():
    a = dense_matrix(6, 5).to_sparse_csr()
    w = rand(5, 3).requires_grad_()
    x = rand(4, 6).requires_grad_()
    assert torch.autograd.gradcheck(
        lambda w_: scorch.matmul(a, w_), (w,), check_forward_ad=True
    )
    assert torch.autograd.gradcheck(
        lambda x_: scorch.matmul(x_, a), (x,), check_forward_ad=True
    )


def test_forward_mode_ad_runs_under_no_grad():
    """Forward-mode AD does not depend on grad mode, so a tangent must survive a call made
    under no_grad -- and the C entry point must not serve such a call from a plan."""
    d = dense_matrix(6, 5)
    a = d.to_sparse_csr()
    w = rand(5, 3)
    for _ in range(3):
        scorch.matmul(a, w)  # a plan for this product
    assert plans_of(a)
    tangent = rand(5, 3, seed=3)
    with fwAD.dual_level(), torch.no_grad():
        out = scorch.matmul(a, fwAD.make_dual(w, tangent))
        torch.testing.assert_close(fwAD.unpack_dual(out).tangent, d @ tangent)


# --------------------------------------------------------------------------- #
# Dispatch: plans, grad mode, parameters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["torch_csr", "stensor_csr"])
def test_a_planned_operand_still_records_the_gradient(kind):
    """Once the operand has a plan, the C entry point serves no-grad calls from it. A call
    that needs a gradient must not be: the plan's result carries no grad_fn."""
    d = dense_matrix()
    a = OPERANDS[kind][0](d)
    w = rand(5, 3)
    for _ in range(3):
        scorch.matmul(a, w)
    assert plans_of(a), "precondition: the operand has a plan"
    w.requires_grad_()
    out = scorch.matmul(a, w)
    assert out.grad_fn is not None
    out.sum().backward()
    torch.testing.assert_close(w.grad, d.t() @ torch.ones(7, 3, dtype=d.dtype))


def test_no_grad_and_inference_mode_record_nothing_and_still_plan():
    a = dense_matrix().to_sparse_csr()
    w = rand(5, 3).requires_grad_()
    with torch.no_grad():
        for _ in range(3):
            assert scorch.matmul(a, w).grad_fn is None
    assert plans_of(a), "a no-grad call must still reach the planned path"
    with torch.inference_mode():
        assert scorch.matmul(a, w).grad_fn is None


@pytest.mark.parametrize("kind", sorted(OPERANDS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("x_shape", ["matrix", "vector", "transposed view"])
def test_dense_times_sparse_is_one_function_in_every_grad_mode(kind, dtype, x_shape):
    """x @ A has the same type and the same values whether a gradient is wanted through it
    (model.train()), not wanted (model.eval()), or switched off by no_grad or
    inference_mode: a dense torch.Tensor every time, as torch.mm returns. It used to be an
    STensor whenever autograd was not involved."""
    d = dense_matrix(dtype=dtype)
    a = OPERANDS[kind][0](d)
    x = {
        "matrix": lambda: rand(4, 7, dtype=dtype),
        "vector": lambda: rand(7, dtype=dtype),
        "transposed view": lambda: rand(7, 4, dtype=dtype).t(),
    }[x_shape]()
    plain = scorch.matmul(x, a)
    xg = x.detach().clone().requires_grad_()
    if x_shape == "transposed view":
        xg = x.t().detach().clone().t().requires_grad_()
    recorded = scorch.matmul(xg, a)
    with torch.no_grad():
        no_grad = scorch.matmul(xg, a)
    with torch.inference_mode():
        inference = scorch.matmul(x, a)
    for out in (plain, recorded, no_grad, inference):
        assert type(out) is torch.Tensor and out.layout is torch.strided
    assert recorded.grad_fn is not None and no_grad.grad_fn is None
    for out in (recorded.detach(), no_grad, inference):
        assert torch.equal(out, plain)
    torch.testing.assert_close(plain, x @ d, **tolerance(dtype))


@pytest.mark.parametrize("kind", sorted(OPERANDS))
def test_an_operand_made_under_inference_mode(kind):
    """Inference tensors have no version counter, and reading one raises: the cached
    structure's stamp reads it as 0 (storage.tensor_version), so x @ A runs under
    inference_mode on an operand made there, and outside it when no gradient is wanted.
    Asking a gradient through such an operand raises torch's own error, as torch.mm does.
    """
    d = dense_matrix()
    with torch.inference_mode():
        a = OPERANDS[kind][0](d)
        x = rand(4, 7)
        for _ in range(2):
            torch.testing.assert_close(scorch.matmul(x, a), x @ d)
    x = rand(4, 7)
    torch.testing.assert_close(scorch.matmul(x, a), x @ d)  # outside, as torch.mm runs
    with pytest.raises(RuntimeError, match="Inference tensors cannot be saved"):
        scorch.matmul(rand(4, 7).requires_grad_(), a)


def test_a_parameter_gets_its_gradient():
    d = dense_matrix()
    a = d.to_sparse_csr()
    w = torch.nn.Parameter(rand(5, 3))
    scorch.matmul(a, w).sum().backward()
    torch.testing.assert_close(w.grad, d.t() @ torch.ones(7, 3, dtype=d.dtype))


# --------------------------------------------------------------------------- #
# The C entry's own record of a planned product (csrc/planned_matmul.h)
# --------------------------------------------------------------------------- #

C_NODE = "PlannedMatmulGrad"


def _planned(a, w):
    """`a` with a plan for products with `w`'s shape: two calls make one (plan.install)."""
    with torch.no_grad():
        for _ in range(3):
            scorch.matmul(a, w)
    assert plans_of(a), "precondition: the operand has a plan"
    return a


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("wants", ["a", "w", "both"])
def test_the_c_entry_records_a_planned_product(dtype, wants):
    d = dense_matrix(dtype=dtype)
    a = _planned(d.to_sparse_csr(), rand(5, 3, dtype=dtype))
    w = rand(5, 3, dtype=dtype)
    a.requires_grad_(wants in ("a", "both"))
    w.requires_grad_(wants in ("w", "both"))
    out = scorch.matmul(a, w)
    assert C_NODE in out.grad_fn.name()
    g = rand(7, 3, dtype=dtype, seed=5)
    out.backward(g)
    ref_a = d.to_sparse_csr().requires_grad_(a.requires_grad)
    ref_w = w.detach().clone().requires_grad_(w.requires_grad)
    ref = torch.sparse.mm(ref_a, ref_w)
    torch.testing.assert_close(out.detach(), ref.detach(), **tolerance(dtype))
    ref.backward(g)
    if w.requires_grad:
        torch.testing.assert_close(w.grad, ref_w.grad, **tolerance(dtype))
    else:
        assert w.grad is None
    if a.requires_grad:
        assert a.grad.layout == torch.sparse_csr
        assert torch.equal(a.grad.crow_indices(), ref_a.grad.crow_indices())
        assert torch.equal(a.grad.col_indices(), ref_a.grad.col_indices())
        torch.testing.assert_close(
            a.grad.values(), ref_a.grad.values(), **tolerance(dtype)
        )
    else:
        assert a.grad is None


def test_the_c_entry_s_product_differentiates_twice():
    """Its backward records its gradients when asked to build a graph, so the second
    derivative is the Python path's: gradcheck and gradgradcheck through a planned operand.
    """
    a = _planned(dense_matrix().to_sparse_csr(), rand(5, 3))
    w = rand(5, 3).requires_grad_()
    assert C_NODE in scorch.matmul(a, w).grad_fn.name()
    assert torch.autograd.gradcheck(lambda w_: scorch.matmul(a, w_), (w,))
    assert torch.autograd.gradgradcheck(lambda w_: scorch.matmul(a, w_) ** 2, (w,))
    # Across operands: d/dA of a gradient taken with respect to w, against dense autograd
    # masked to A's pattern (A's gradient is masked; torch.sparse.mm's second one is not).
    a.requires_grad_()
    (gw,) = torch.autograd.grad(
        scorch.matmul(a, w).pow(2).sum(), (w,), create_graph=True
    )
    (second,) = torch.autograd.grad(gw.sum(), (a,))
    dense = a.detach().to_dense().requires_grad_()
    (ref_gw,) = torch.autograd.grad((dense @ w).pow(2).sum(), (w,), create_graph=True)
    (ref_second,) = torch.autograd.grad(ref_gw.sum(), (dense,))
    torch.testing.assert_close(second.to_dense(), ref_second * (dense != 0))


def test_the_c_entry_checks_saved_values_like_torch():
    """A's values are saved for b's gradient, so writing them in place between the forward
    and the backward is an error, as it is for torch.sparse.mm and the Python path."""
    a = _planned(dense_matrix().to_sparse_csr(), rand(5, 3))
    w = rand(5, 3).requires_grad_()
    out = scorch.matmul(a, w)
    assert C_NODE in out.grad_fn.name()
    with torch.no_grad():
        a.values().mul_(2.0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


def test_an_error_in_the_c_entry_s_backward_arrives_as_itself(monkeypatch):
    class Marker(Exception):
        pass

    def fail(*args):
        raise Marker("from the backward")

    a = _planned(dense_matrix().to_sparse_csr(), rand(5, 3))
    out = scorch.matmul(a, rand(5, 3).requires_grad_())
    assert C_NODE in out.grad_fn.name()
    monkeypatch.setattr(grad_mod, "_mm_backward", fail)
    with pytest.raises(Marker, match="from the backward"):
        out.sum().backward()


def test_the_c_entry_s_record_holds_its_operand_only_as_long_as_the_graph():
    a = _planned(dense_matrix().to_sparse_csr(), rand(5, 3)).requires_grad_()
    w = rand(5, 3).requires_grad_()
    out = scorch.matmul(a, w)
    assert C_NODE in out.grad_fn.name()
    ref = weakref.ref(a)
    del a
    assert ref() is not None, "the graph still holds the operand"
    out.sum().backward()
    del out
    gc.collect()
    assert ref() is None


def test_the_python_path_keeps_what_the_c_entry_does_not_record():
    """A transposed dense operand, a forward-mode tangent, and a withdrawn backward each
    go to the Python function, which records them as before."""
    d = dense_matrix()
    a = _planned(d.to_sparse_csr(), rand(5, 3))
    w_t = rand(3, 5).requires_grad_()
    out = scorch.matmul(a, w_t.t())
    assert C_NODE not in out.grad_fn.name()
    out.sum().backward()
    torch.testing.assert_close(w_t.grad, (d.t() @ torch.ones(7, 3, dtype=d.dtype)).t())

    w = rand(5, 3).requires_grad_()
    tangent = rand(5, 3, seed=3)
    with fwAD.dual_level():
        out = scorch.matmul(a, fwAD.make_dual(w, tangent))
        assert C_NODE not in out.grad_fn.name()
        torch.testing.assert_close(fwAD.unpack_dual(out).tangent, d @ tangent)

    scorch_ops.set_planned_matmul_backward(None)
    try:
        assert C_NODE not in scorch.matmul(a, w).grad_fn.name()
    finally:
        scorch_ops.set_planned_matmul_backward(grad_mod.planned_mm_backward)
    assert C_NODE in scorch.matmul(a, w).grad_fn.name()


# --------------------------------------------------------------------------- #
# x @ A served, and recorded, by the C entry (csrc/planned_matmul.h)
# --------------------------------------------------------------------------- #

C_NODE_T = "torch::autograd::CppNode<scorch_native::PlannedMatmulGradT>"


def _right_node(out):
    """The node that recorded x @ A: the output's own, since the Function (C or Python) returns
    x @ A itself, not a product the caller receives transposed, and it is not a view."""
    assert out._base is None
    return out.grad_fn.name()


def _served(a):
    return sum(tp.served for tp in plan_mod.transposed_plans_of(a).values())


def _restore_right():
    scorch_ops.set_planned_matmul_right(
        grad_mod.planned_mm_t_backward, plan_mod.TRANSPOSED_PLANS_ATTR
    )


def _planned_right(a, rows):
    """`a` whose x @ A the C entry serves for an x of `rows` rows: two calls give the
    transpose's carrier a plan (plan.install), and the next records it in `a`'s state."""
    x = rand(rows, a.shape[0], dtype=a.dtype, seed=9)
    with torch.no_grad():
        for _ in range(3):
            scorch.matmul(x, a)
    assert plan_mod.transposed_plans_of(a), "precondition: the operand records its x @ A"
    return a


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("wants", ["x", "a", "both"])
def test_the_c_entry_records_x_times_a(dtype, rows, wants):
    d = dense_matrix(dtype=dtype)
    a = _planned_right(d.to_sparse_csr(), rows)
    x = rand(rows, 7, dtype=dtype).requires_grad_(wants in ("x", "both"))
    a.requires_grad_(wants in ("a", "both"))
    before = _served(a)
    out = scorch.matmul(x, a)
    assert _right_node(out) == C_NODE_T and _served(a) == before + 1
    assert out.shape == (rows, 5) and out.is_contiguous()
    g = rand(rows, 5, dtype=dtype, seed=5)
    out.backward(g)
    d_ref = d.clone().requires_grad_(a.requires_grad)
    x_ref = x.detach().clone().requires_grad_(x.requires_grad)
    ref = x_ref @ d_ref
    ref.backward(g)
    tol = tolerance(dtype)
    torch.testing.assert_close(out.detach(), ref.detach(), **tol)
    if x.requires_grad:
        torch.testing.assert_close(x.grad, x_ref.grad, **tol)
    else:
        assert x.grad is None
    if a.requires_grad:
        assert a.grad.layout == torch.sparse_csr
        torch.testing.assert_close(a.grad.to_dense(), d_ref.grad * (d != 0), **tol)
    else:
        assert a.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rows", [1, 3, 16])
def test_the_c_entry_s_x_times_a_is_the_python_function_s_bit_for_bit(dtype, rows):
    """The same kernels on the same arrays: the product and both gradients equal the Python
    function's exactly, recorded, unrecorded, and under inference_mode."""
    d = dense_matrix(rows=40, cols=30, density=0.2, dtype=dtype)
    a = _planned_right(d.to_sparse_csr(), rows).requires_grad_()
    x = rand(rows, 40, dtype=dtype).requires_grad_()
    g = rand(rows, 30, dtype=dtype, seed=4)

    def run():
        x.grad = a.grad = None
        out = scorch.matmul(x, a)
        out.backward(g)
        with torch.no_grad():
            plain = scorch.matmul(x, a)
        with torch.inference_mode():
            inference = scorch.matmul(x.detach(), a)
        return [out.detach(), x.grad, a.grad.values(), plain, inference], _right_node(out)

    before = _served(a)
    served, node = run()
    assert node == C_NODE_T and _served(a) == before + 3
    scorch_ops.set_planned_matmul_right(None)
    try:
        python, python_node = run()
    finally:
        _restore_right()
    assert python_node != C_NODE_T and _served(a) == before + 3
    for got, want in zip(served, python):
        assert torch.equal(got, want)


def test_x_times_a_goes_to_the_python_function_where_the_c_entry_does_not_serve_it():
    """A transposed view, a vector, a Parameter, a forward-mode tangent and a withdrawn entry
    are each computed by the Python function, correctly, without touching the record."""
    d = dense_matrix()
    a = _planned_right(d.to_sparse_csr(), 4)
    served = _served(a)
    tol = tolerance(d.dtype)
    base = rand(7, 4).requires_grad_()
    out = scorch.matmul(base.t(), a)
    assert _right_node(out) != C_NODE_T
    out.sum().backward()
    torch.testing.assert_close(out.detach(), base.detach().t() @ d, **tol)
    torch.testing.assert_close(base.grad, (torch.ones(4, 5, dtype=d.dtype) @ d.t()).t(), **tol)
    v = rand(7).requires_grad_()
    torch.testing.assert_close(scorch.matmul(v, a).detach(), v.detach() @ d, **tol)
    p = torch.nn.Parameter(rand(4, 7))
    out = scorch.matmul(p, a)
    assert _right_node(out) != C_NODE_T
    out.sum().backward()
    torch.testing.assert_close(p.grad, torch.ones(4, 5, dtype=d.dtype) @ d.t(), **tol)
    x = rand(4, 7).requires_grad_()
    tangent = rand(4, 7, seed=3)
    with fwAD.dual_level():
        out = scorch.matmul(fwAD.make_dual(x, tangent), a)
        torch.testing.assert_close(fwAD.unpack_dual(out).tangent, tangent @ d, **tol)
    assert _served(a) == served
    assert _right_node(scorch.matmul(x, a)) == C_NODE_T
    assert _served(a) == served + 1


@pytest.mark.parametrize("write", ["col_indices", "crow_indices", "values"])
def test_x_times_a_is_not_served_from_an_operand_written_in_place(write):
    """One nonzero per row, then a write through one of the operand's aliases. Every alias
    shares the operand's version counter, so the record no longer describes the operand: the
    next x @ A goes to the Python function, which computes the new product, and the stale
    record is dropped. A later served product records afresh."""
    rows, cols = 6, 5
    crow = torch.arange(rows + 1)
    col = torch.tensor([0, 1, 2, 3, 4, 0])
    a = torch.sparse_csr_tensor(crow, col, rand(rows), (rows, cols))
    a = _planned_right(a, 3)
    x = rand(3, rows).requires_grad_()
    assert _right_node(scorch.matmul(x, a)) == C_NODE_T
    with torch.no_grad():
        if write == "col_indices":
            a.col_indices().copy_(torch.tensor([4, 3, 2, 1, 0, 4]))
        elif write == "crow_indices":  # rows 0 and 1 swap their one nonzero for two and none
            a.crow_indices()[1] = 2
        else:
            a.values().mul_(3.0)
    out = scorch.matmul(x, a)
    assert _right_node(out) != C_NODE_T
    assert not plan_mod.transposed_plans_of(a), "the stale record was dropped"
    torch.testing.assert_close(out.detach(), x.detach() @ a.to_dense(), **tolerance(x.dtype))
    _planned_right(a, 3)
    out = scorch.matmul(x, a)
    assert _right_node(out) == C_NODE_T
    torch.testing.assert_close(out.detach(), x.detach() @ a.to_dense(), **tolerance(x.dtype))


def test_the_c_entry_s_x_times_a_checks_saved_tensors_like_torch():
    """x^T is saved for A's gradient and A's values for x's, so writing either in place
    between the forward and the backward is an error, as on the Python path. (With A
    requiring grad too, a write to its values fails the backward with a different message,
    on every path; see test_a_values_write_fails_the_backward_on_every_path.)"""
    a = _planned_right(dense_matrix().to_sparse_csr(), 4).requires_grad_()
    x = rand(4, 7).requires_grad_()
    out = scorch.matmul(x, a)
    assert _right_node(out) == C_NODE_T
    with torch.no_grad():
        x.mul_(2.0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()
    a.requires_grad_(False)
    out = scorch.matmul(x, a)
    assert _right_node(out) == C_NODE_T
    with torch.no_grad():
        a.values().mul_(2.0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


@pytest.mark.parametrize("path", ["c entry", "python"])
@pytest.mark.parametrize("side", ["A @ B", "x @ A"])
def test_a_values_write_fails_the_backward_on_every_path(path, side):
    """A requiring grad, its values written in place between the forward and the backward:
    the backward fails, on every path. torch.sparse.mm reports "modified by an inplace
    operation"; scorch's paths save A.values(), a view of A, and torch fails regenerating
    that view's history instead ("Sparse CSR tensors do not have strides"). The message
    differs from torch's; what matters here is that no gradient is computed."""
    d = dense_matrix()
    if side == "A @ B":
        a = _planned(d.to_sparse_csr(), rand(5, 3))
        dense = rand(5, 3).requires_grad_()
        run = lambda: scorch.matmul(a, dense)  # noqa: E731
    else:
        a = _planned_right(d.to_sparse_csr(), 4)
        dense = rand(4, 7).requires_grad_()
        run = lambda: scorch.matmul(dense, a)  # noqa: E731
    a.requires_grad_()
    if path == "python":
        scorch_ops.set_planned_matmul_backward(None)
        scorch_ops.set_planned_matmul_right(None)
    try:
        out = run()
        node = out.grad_fn.name() if side == "A @ B" else _right_node(out)
        assert ("PlannedMatmulGrad" in node) == (path == "c entry")
        with torch.no_grad():
            a.values().mul_(2.0)
        with pytest.raises(RuntimeError):
            out.sum().backward()
        assert dense.grad is None
    finally:
        scorch_ops.set_planned_matmul_backward(grad_mod.planned_mm_backward)
        _restore_right()


def test_the_c_entry_s_x_times_a_differentiates_twice():
    a = _planned_right(dense_matrix().to_sparse_csr(), 3)
    x = rand(3, 7).requires_grad_()
    assert _right_node(scorch.matmul(x, a)) == C_NODE_T
    served = _served(a)
    assert torch.autograd.gradcheck(lambda x_: scorch.matmul(x_, a), (x,))
    assert torch.autograd.gradgradcheck(lambda x_: scorch.matmul(x_, a) ** 2, (x,))
    assert _served(a) > served, "gradcheck went through the C entry"


def test_the_c_entry_s_x_times_a_leaves_the_shared_transpose_s_buffer_alone():
    """Operands over the same index arrays share autograd.py's transpose, and with it the
    values buffer of its carrier, whose writers serialize on the structure's lock (held across
    gather_values, which releases the GIL). The C entry does not take that lock: it gathers
    into its record's own buffer, so an x @ A it serves leaves the carrier's as the last
    product through it left it."""
    d = dense_matrix(rows=40, cols=30, density=0.2)
    base = d.to_sparse_csr()
    crow, col = base.crow_indices(), base.col_indices()
    a1 = torch.sparse_csr_tensor(crow, col, base.values().clone(), d.shape)
    a1 = _planned_right(a1, 3)
    a2 = torch.sparse_csr_tensor(crow, col, base.values() * 2.0, d.shape)
    x = rand(3, 40)
    torch.testing.assert_close(scorch.matmul(x, a2), x @ (2.0 * d), **tolerance(d.dtype))
    structure, _ = grad_mod._torch_operand(a2)[0].stored.transpose()
    assert structure is grad_mod._torch_operand(a1)[0].stored.transpose()[0]
    carrier_buffer = structure._carriers[d.dtype][1]
    snapshot = carrier_buffer.clone()
    before = _served(a1)
    out = scorch.matmul(x, a1)
    assert _served(a1) == before + 1
    torch.testing.assert_close(out, x @ d, **tolerance(d.dtype))
    assert torch.equal(carrier_buffer, snapshot)
    for tp in plan_mod.transposed_plans_of(a1).values():
        assert tp.buffer.data_ptr() != carrier_buffer.data_ptr()


def test_an_operand_s_records_of_one_transpose_share_one_buffer():
    a = dense_matrix(rows=40, cols=30, density=0.2).to_sparse_csr()
    _planned_right(a, 3)
    _planned_right(a, 5)
    first, second = plan_mod.transposed_plans_of(a).values()
    assert {first.free_dim, second.free_dim} == {3, 5}
    assert first.buffer.data_ptr() == second.buffer.data_ptr()


def test_x_times_a_s_record_lives_with_its_operand():
    a = _planned_right(dense_matrix().to_sparse_csr(), 2)
    plan_mod.forget(a)
    assert not plan_mod.transposed_plans_of(a)
    a = _planned_right(a, 2)
    key = id(a)
    del a
    gc.collect()
    assert key not in plan_mod.TORCH_OPERANDS


# --------------------------------------------------------------------------- #
# The cached structure
# --------------------------------------------------------------------------- #


def test_the_transpose_is_built_once_per_structure(monkeypatch):
    calls = []
    real = grad_mod._native.csr_transpose

    class Counting:
        def __getattr__(self, name):
            return getattr(scorch_ops, name)

        @staticmethod
        def csr_transpose(*args):
            calls.append(1)
            return real(*args)

    monkeypatch.setattr(grad_mod, "_native", Counting())
    a = dense_matrix().to_sparse_csr()
    w = rand(5, 3).requires_grad_()
    for _ in range(4):
        scorch.matmul(a, w).sum().backward()
    assert len(calls) == 1


def test_values_changed_in_place_reach_the_transposed_product():
    """A training step changes A's values in place: the next backward pass must see the
    new values, not the ones the transpose's buffer was last filled with."""
    d = dense_matrix()
    a = d.to_sparse_csr().requires_grad_()
    w = rand(5, 3).requires_grad_()
    for scale in (1.0, 3.0, -2.0):
        with torch.no_grad():
            a.values().copy_(d.to_sparse_csr().values() * scale)
        w.grad = None
        a.grad = None
        scorch.matmul(a, w).sum().backward()
        torch.testing.assert_close(
            w.grad, (d * scale).t() @ torch.ones(7, 3, dtype=d.dtype)
        )


@pytest.mark.parametrize("fresh", [False, True], ids=["one operand", "fresh operands"])
def test_values_written_through_data_reach_the_product(fresh):
    """A write through .data does not move the version counter -- gradcheck perturbs its
    inputs that way, and older optimizers step that way -- so the copy of the values the
    transposed product runs on is refilled on every call rather than when a stamp moves.
    """
    d = dense_matrix()
    crow, col = d.to_sparse_csr().crow_indices(), d.to_sparse_csr().col_indices()
    values = d.to_sparse_csr().values().clone()
    a = torch.sparse_csr_tensor(crow, col, values, d.shape)
    x = rand(4, 7)
    for scale in (1.0, 3.0, -2.0):
        values.data.copy_(d.to_sparse_csr().values() * scale)
        if fresh:
            a = torch.sparse_csr_tensor(crow, col, values, d.shape)
        torch.testing.assert_close(scorch.matmul(x, a), x @ (d * scale))


def _transposes_over(*arrays):
    ids = {id(t.untyped_storage()) for t in arrays}
    return [k for k in grad_mod._TRANSPOSES if any(part[0] in ids for part in k[2:])]


def test_fresh_operands_over_the_same_arrays_share_one_transpose(monkeypatch):
    """A new operand around the same index arrays on every call -- values that change, say
    -- finds the transpose built for an earlier one."""
    calls = []
    real = grad_mod._native.csr_transpose

    class Counting:
        def __getattr__(self, name):
            return getattr(scorch_ops, name)

        @staticmethod
        def csr_transpose(*args):
            calls.append(1)
            return real(*args)

    monkeypatch.setattr(grad_mod, "_native", Counting())
    d = dense_matrix()
    crow, col = d.to_sparse_csr().crow_indices(), d.to_sparse_csr().col_indices()
    values = d.to_sparse_csr().values()
    x = rand(4, 7)
    for scale in (1.0, 2.0, 3.0):
        a = torch.sparse_csr_tensor(crow, col, values * scale, d.shape)
        torch.testing.assert_close(scorch.matmul(x, a), x @ (d * scale))
    assert len(calls) == 1


def test_the_shared_transpose_lives_as_long_as_its_arrays():
    """An entry is dropped when the arrays it describes die, and an array changed in place
    replaces its entry rather than adding one."""
    d = dense_matrix()
    crow = d.to_sparse_csr().crow_indices().clone()
    col = d.to_sparse_csr().col_indices().clone()
    x = rand(4, 7)
    a = torch.sparse_csr_tensor(crow, col, d.to_sparse_csr().values(), d.shape)
    scorch.matmul(x, a)
    assert len(_transposes_over(crow, col)) == 1
    with torch.no_grad():
        col.copy_(col)  # moves the version counter, contents unchanged
    a = torch.sparse_csr_tensor(crow, col, d.to_sparse_csr().values(), d.shape)
    torch.testing.assert_close(scorch.matmul(x, a), x @ d)
    assert len(_transposes_over(crow, col)) == 1
    before = len(grad_mod._TRANSPOSES)
    del a, crow, col
    gc.collect()
    assert len(grad_mod._TRANSPOSES) == before - 1


def test_an_operand_whose_index_arrays_were_replaced_gets_a_new_structure():
    a = torch.eye(4, dtype=torch.float64).to_sparse_csr()
    w = rand(4, 3).requires_grad_()
    scorch.matmul(a, w).sum().backward()
    target = torch.eye(4, dtype=torch.float64).flip(1).to_sparse_csr()
    a.resize_as_sparse_(torch.ones(4, 4, dtype=torch.float64).to_sparse_csr())
    a.resize_as_sparse_(target)
    a.copy_(target)
    w.grad = None
    g = rand(4, 3, seed=4)
    (scorch.matmul(a, w) * g).sum().backward()
    torch.testing.assert_close(w.grad, target.to_dense().t() @ g)


@pytest.mark.parametrize("layout", ["csr", "coo"])
@pytest.mark.parametrize("sparse_grad", [False, True])
def test_the_structure_dies_with_its_torch_operand(layout, sparse_grad):
    """The structure lives in the operand's own registry entry, so nothing it holds may
    refer back to the operand: values() of a CSR tensor and indices() of a COO tensor are
    views whose base is the tensor, and holding either kept every trained matrix alive.
    """
    d = dense_matrix()
    a = d.to_sparse_csr() if layout == "csr" else d.to_sparse_coo()
    a.requires_grad_(sparse_grad)
    w = rand(5, 3).requires_grad_()
    for _ in range(2):
        # Reset before each backward pass: accumulating a second sparse CSR gradient
        # needs a torch built with MKL, which the ARM host's is not.
        a.grad = None
        scorch.matmul(a, w).sum().backward()
        a.grad = None
        scorch.matmul(rand(4, 7).requires_grad_(), a).sum().backward()  # x @ A
    key = id(a)
    assert plan_mod.GRAD_ATTR in plan_mod.TORCH_OPERANDS[key][1]
    del a
    assert key not in plan_mod.TORCH_OPERANDS


def test_an_stensor_with_a_structure_still_pickles():
    s = STensor.from_torch(dense_matrix().to_sparse_csr())
    w = rand(5, 3).requires_grad_()
    scorch.matmul(s, w).sum().backward()
    assert plan_mod.GRAD_ATTR in s.__dict__
    clone = pickle.loads(pickle.dumps(s))
    assert plan_mod.GRAD_ATTR not in clone.__dict__


@pytest.mark.parametrize("side", ["left", "right"])
def test_invalid_index_arrays_fail_closed(side):
    """torch checks nothing about a CSR tensor's arrays by default; the structure the
    gradient kernels read must be screened before they index with it. x @ A runs its
    forward pass on the transpose, so there nothing else would have looked first."""
    crow = torch.tensor([0, 1, 2])
    col = torch.tensor([0, 9])  # column 9 of a 2 x 3 matrix
    a = torch.sparse_csr_tensor(crow, col, torch.ones(2, dtype=torch.float64), (2, 3))
    with pytest.raises(TensorIndexError):
        if side == "left":
            scorch.matmul(a, rand(3, 2).requires_grad_())
        else:
            scorch.matmul(rand(4, 2).requires_grad_(), a)


# --------------------------------------------------------------------------- #
# A recorded product's output is never a view (patch 0082)
# --------------------------------------------------------------------------- #
#
# torch takes a custom Function's output that is a view made inside the Function (or an
# input returned as-is) for a view it cannot replay: an in-place operation on it in grad
# mode raises, and so does reading its grad_fn once it or its base was written in place
# under no_grad.  A training loop does both: nn.ReLU(inplace=True) after a product, and an
# optimizer step on a tensor that aliases the product.  Every Function that records a
# product returns a tensor that is not a view, so each write below is allowed, and the
# gradient is the one dense float64 torch computes for the same write.

WRITES = ["relu_ in grad mode", "add_ under no_grad"]


def _write(write, *tensors):
    """The same in-place write on each tensor: relu_ recorded, as nn.ReLU(inplace=True)
    does, or add_ under no_grad, as an optimizer step does, after which reading grad_fn
    is what torch refused for a view made inside a Function."""
    for t in tensors:
        if write == "relu_ in grad mode":
            t.relu_()
        else:
            with torch.no_grad():
                t.add_(1.0)
            if t.requires_grad:
                assert t.grad_fn is not None


# path: (side, how the dense operand is laid out, what records the product, its node).
# A transposed dense operand and a vector go to autograd.py's Functions, each of which returns
# the product in the shape the caller receives (x @ A, a vector), so the caller's tensor is the
# Function's output itself; the C entry records the two contiguous matrix products of a
# planned operand.
RECORDED_PRODUCTS = {
    "A @ b": ("left", "matrix", "python", "_SparseMMBackward"),
    "A @ b.t()": ("left", "transposed", "python", "_SparseMMBackward"),
    "A @ v": ("left", "vector", "python", "_SparseMMBackward"),
    "x @ A": ("right", "matrix", "python", "_SparseXABackward"),
    "x.t() @ A": ("right", "transposed", "python", "_SparseXABackward"),
    "v @ A": ("right", "vector", "python", "_SparseMMTBackward"),
    "A @ b, C entry": ("left", "matrix", "c entry", C_NODE),
    "x @ A, C entry": ("right", "matrix", "c entry", C_NODE_T),
}


@contextlib.contextmanager
def _recorded_by(entry):
    """For "python", the C entry's recording turned off, so a Function in autograd.py
    records the product even if the operand had a plan."""
    if entry == "c entry":
        yield
        return
    scorch_ops.set_planned_matmul_backward(None)
    scorch_ops.set_planned_matmul_right(None)
    try:
        yield
    finally:
        scorch_ops.set_planned_matmul_backward(grad_mod.planned_mm_backward)
        _restore_right()


def _recording_node(out):
    """The node that recorded the product: the output's own, since the caller receives the
    Function's output itself, never a view of it (a transpose or a squeeze)."""
    return out.grad_fn.name()


def _recorded_product(path, dtype, wants):
    """The product ``path`` names, recorded, with the leaves a gradient is asked of, and the
    same product in dense float64 torch: (out, a, raw, ref, a64, raw64), where ``raw`` is
    the dense leaf (the operand is its transpose for a transposed layout)."""
    side, layout, entry, node = RECORDED_PRODUCTS[path]
    left = side == "left"
    d = dense_matrix(dtype=dtype)
    inner = d.shape[1] if left else d.shape[0]
    shape = {
        "matrix": (inner, 3) if left else (4, inner),
        "transposed": (3, inner) if left else (inner, 4),
        "vector": (inner,),
    }[layout]
    raw = rand(*shape, dtype=dtype) - 0.5  # entries of both signs, for relu_ to change
    a = d.to_sparse_csr()
    if entry == "c entry":
        a = _planned(a, rand(inner, 3, dtype=dtype)) if left else _planned_right(a, 4)
    a.requires_grad_(wants in ("a", "both"))
    raw.requires_grad_(wants in ("dense", "both"))
    dense = raw.t() if layout == "transposed" else raw
    with _recorded_by(entry):
        out = scorch.matmul(a, dense) if left else scorch.matmul(dense, a)
    assert node in _recording_node(out)
    # the product the caller receives is never a view, as torch.matmul's is not
    assert out._base is None

    a64 = d.double().requires_grad_(a.requires_grad)
    raw64 = raw.detach().double().requires_grad_(raw.requires_grad)
    dense64 = raw64.t() if layout == "transposed" else raw64
    ref = a64 @ dense64 if left else dense64 @ a64
    return out, a, raw, ref, a64, raw64


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("wants", ["a", "dense", "both"])
@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize("path", list(RECORDED_PRODUCTS))
def test_a_recorded_products_output_takes_in_place_writes(path, write, wants, dtype):
    out, a, raw, ref, a64, raw64 = _recorded_product(path, dtype, wants)
    _write(write, out, ref)
    g = rand(*ref.shape, seed=2)
    (out * g.to(dtype)).sum().backward()
    (ref * g).sum().backward()

    tol = tolerance(dtype)
    torch.testing.assert_close(out.detach().double(), ref.detach(), **tol)
    if raw.requires_grad:
        torch.testing.assert_close(raw.grad.double(), raw64.grad, **tol)
    else:
        assert raw.grad is None
    if a.requires_grad:
        assert a.grad.layout == torch.sparse_csr
        mask = a64.detach() != 0
        torch.testing.assert_close(a.grad.to_dense().double(), a64.grad * mask, **tol)
    else:
        assert a.grad is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("path", list(RECORDED_PRODUCTS))
def test_a_recorded_products_gradient_takes_in_place_writes_and_differentiates_again(
    path, dtype
):
    """With create_graph=True the backward records each gradient as a product too
    (_SparseMMT for the dense operand's of A @ b, _SparseMM for that of x @ A), so the same
    holds one order up: relu_ on the dense operand's gradient in grad mode, then that
    gradient's derivative with respect to both operands, against dense float64 torch."""
    out, a, raw, ref, a64, raw64 = _recorded_product(path, dtype, "both")
    (gd,) = torch.autograd.grad(out.pow(2).sum(), (raw,), create_graph=True)
    (gd64,) = torch.autograd.grad(ref.pow(2).sum(), (raw64,), create_graph=True)
    gd.relu_()
    gd64.relu_()
    h = rand(*gd64.shape, seed=6)
    ga, graw = torch.autograd.grad((gd * h.to(dtype)).sum(), (a, raw))
    ga64, graw64 = torch.autograd.grad((gd64 * h).sum(), (a64, raw64))

    tol = tolerance(dtype)
    torch.testing.assert_close(gd.detach().double(), gd64.detach(), **tol)
    torch.testing.assert_close(graw.double(), graw64, **tol)
    mask = a64.detach() != 0
    torch.testing.assert_close(ga.to_dense().double(), ga64 * mask, **tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("write", WRITES)
def test_the_sddmm_functions_output_takes_in_place_writes(write, dtype):
    """_SparseSDDMM records the gradient of A's values when a backward builds a graph; its
    output is those values, which a second-order method may write in place.  Called
    directly here, against the same sampled product in dense float64 torch."""
    d = dense_matrix(dtype=dtype)
    a = d.to_sparse_csr()
    op, _ = grad_mod._torch_operand(a)
    x = (rand(7, 3, dtype=dtype) - 0.5).requires_grad_()
    y = (rand(5, 3, dtype=dtype, seed=3) - 0.5).requires_grad_()
    out = grad_mod._SparseSDDMM.apply(op, x, y)
    assert out.grad_fn.name() == "_SparseSDDMMBackward"

    rows = torch.repeat_interleave(torch.arange(7), a.crow_indices().diff())
    cols = a.col_indices()
    x64 = x.detach().double().requires_grad_()
    y64 = y.detach().double().requires_grad_()
    ref = (x64[rows] * y64[cols]).sum(1)
    _write(write, out, ref)
    g = rand(*ref.shape, seed=4)
    (out * g.to(dtype)).sum().backward()
    (ref * g).sum().backward()

    tol = tolerance(dtype)
    torch.testing.assert_close(out.detach().double(), ref.detach(), **tol)
    torch.testing.assert_close(x.grad.double(), x64.grad, **tol)
    torch.testing.assert_close(y.grad.double(), y64.grad, **tol)


# --------------------------------------------------------------------------- #
# Products autograd does not cover
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_generated_kernels_product_has_the_einsums_derivative(dtype):
    """The generated kernel's product is differentiable.  use_cache=False asks for the
    generated kernel, which the SpMM Functions above do not cover; the product is the
    einsum ij,jk->ik and is recorded with that einsum's derivative (einsum_grad), for
    both operands.  Before step 3 its backward raised matmul_derivative_unsupported.
    (use_cache=True is the default, and since patch 0072 the call without it.)"""
    d = dense_matrix(dtype=dtype)
    a = d.to_sparse_csr().requires_grad_()
    w = rand(5, 3, dtype=dtype).requires_grad_()
    out = scorch.matmul(a, w, use_cache=False)
    torch.testing.assert_close(out, d @ w.detach(), **tolerance(dtype))
    assert out.grad_fn is not None
    upstream = rand(7, 3, dtype=dtype, seed=2)
    (out * upstream).sum().backward()

    # The float64 reference, on the values rounded to the key's dtype.
    d64, w64, g64 = d.double(), w.detach().double(), upstream.double()
    torch.testing.assert_close(w.grad.double(), d64.t() @ g64, **tolerance(dtype))
    assert a.grad.layout is torch.sparse_csr
    torch.testing.assert_close(
        a.grad.to_dense().double(), (g64 @ w64.t()) * (d64 != 0), **tolerance(dtype)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("write", WRITES)
def test_the_generated_kernels_product_takes_in_place_writes(write, dtype):
    """The generated kernel's product, recorded with the einsum's derivative, is not a
    view either (see the section above): it takes either write, and both operands'
    gradients are dense float64 torch's for the same write.  Before step 3 this product
    had no derivative, and the test was that its refusal survives the write."""
    d = dense_matrix(dtype=dtype)
    a = d.to_sparse_csr().requires_grad_()
    w = (rand(5, 3, dtype=dtype) - 0.5).requires_grad_()
    out = scorch.matmul(a, w, use_cache=False)
    assert out.grad_fn is not None
    a64 = d.double().requires_grad_()
    w64 = w.detach().double().requires_grad_()
    ref = a64 @ w64
    _write(write, out, ref)
    g = rand(7, 3, seed=2)
    (out * g.to(dtype)).sum().backward()
    (ref * g).sum().backward()

    tol = tolerance(dtype)
    torch.testing.assert_close(out.detach().double(), ref.detach(), **tol)
    torch.testing.assert_close(w.grad.double(), w64.grad, **tol)
    assert a.grad.layout is torch.sparse_csr
    mask = a64.detach() != 0
    torch.testing.assert_close(a.grad.to_dense().double(), a64.grad * mask, **tol)


# What each of autograd.py's Functions returns, seen at the Function itself: never a view
# (torch's rule above) and never one of its inputs, on every operand kind, both layouts of
# the dense operand, and shapes with one row or one column, where ``_transpose`` makes no
# copy.  The caller's tensor is then checked against dense float64 torch and written in
# place.  The Functions make no ``.detach()`` on a matrix's path: the kernels' products are
# tensors of their own (see the note above autograd._x_at_a), so this is what stands behind
# not making one.

# (module, Function): autograd.py's, and einsum_grad's for a product they do not cover
SPIED = (
    (grad_mod, "_SparseMM"),
    (grad_mod, "_SparseMMT"),
    (grad_mod, "_SparseXA"),
    (grad_mod, "_SparseSDDMM"),
    (grad_mod, "_NotDifferentiable"),
    (einsum_grad_mod, "_EinsumValues"),
)


@contextlib.contextmanager
def _function_outputs():
    """Every (Function name, output, tensor inputs) the Functions in SPIED return meanwhile."""
    seen = []
    saved = {name: getattr(mod, name).apply for mod, name in SPIED}
    own = {name: getattr(mod, name).__dict__.get("apply") for mod, name in SPIED}

    def spy(name):
        def apply(*args):
            out = saved[name](*args)
            seen.append((name, out, [x for x in args if isinstance(x, torch.Tensor)]))
            return out

        return staticmethod(apply)

    for mod, name in SPIED:
        setattr(getattr(mod, name), "apply", spy(name))
    try:
        yield seen
    finally:
        for mod, name in SPIED:
            if own[name] is None:  # torch.autograd.Function's apply again
                delattr(getattr(mod, name), "apply")
            else:
                setattr(getattr(mod, name), "apply", own[name])


def _owned(seen, names):
    assert {name for name, _, _ in seen} >= set(names), [name for name, _, _ in seen]
    for name, out, inputs in seen:
        assert out._base is None, f"{name} returned a view"
        assert all(out is not x for x in inputs), f"{name} returned an input"


# (rows, cols) of A, and the dense operand's free extent: one row or one column of A, or of
# the dense operand, is where _transpose returns a view instead of a copy
SHAPES = {
    "7x5, 3": (7, 5, 3),
    "1x5, 3": (1, 5, 3),
    "7x1, 3": (7, 1, 3),
    "7x5, 1": (7, 5, 1),
}


@pytest.mark.parametrize("kind", sorted(OPERANDS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("layout", ["matrix", "transposed"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_no_function_returns_a_view_or_an_input(kind, dtype, side, layout, shape):
    rows, cols, free = SHAPES[shape]
    make, sparse_grad = OPERANDS[kind]
    d = dense_matrix(rows, cols, density=0.8, dtype=dtype)
    a = make(d)
    if sparse_grad:
        a.requires_grad_()
    left = side == "left"
    inner = cols if left else rows
    shaped = (inner, free) if left else (free, inner)
    raw = rand(*(shaped[::-1] if layout == "transposed" else shaped), dtype=dtype) - 0.5
    raw.requires_grad_()
    dense = raw.t() if layout == "transposed" else raw
    with _recorded_by("python"), _function_outputs() as seen:
        first = scorch.matmul(a, dense) if left else scorch.matmul(dense, a)
        # the same product again: the plan the first call installed serves it (the warm call)
        out = scorch.matmul(a, dense) if left else scorch.matmul(dense, a)
        # a graph asked of the backward records the gradients as products too, and the
        # values' gradient of a differentiable sparse operand as _SparseSDDMM
        grads = torch.autograd.grad(
            out.pow(2).sum(), (a, raw) if sparse_grad else (raw,), create_graph=True
        )
    _owned(seen, ["_SparseMM" if left else "_SparseXA"])
    if sparse_grad:
        assert "_SparseSDDMM" in {name for name, _, _ in seen}
    # and what the caller receives is the Function's output itself, not a view of it
    assert first._base is None and out._base is None

    d64 = d.double()
    raw64 = raw.detach().double().requires_grad_()
    dense64 = raw64.t() if layout == "transposed" else raw64
    ref = d64 @ dense64 if left else dense64 @ d64
    tol = tolerance(dtype)
    torch.testing.assert_close(first.detach().double(), ref.detach(), **tol)
    torch.testing.assert_close(out.detach().double(), ref.detach(), **tol)
    (g64,) = torch.autograd.grad(ref.pow(2).sum(), (raw64,))
    torch.testing.assert_close(grads[-1].detach().double(), g64, **tol)
    out.relu_()
    grads[-1].relu_()


@pytest.mark.parametrize("kind", sorted(OPERANDS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("layout", ["matrix", "transposed", "vector"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_a_product_without_a_gradient_is_not_a_view(kind, dtype, side, layout, shape):
    """With no gradient wanted, the product a call returns is a tensor of its own too, as
    torch.matmul's is: on the first call (for a torch CSR operand, the C entry's unplanned
    product; else the prebuilt kernel's route or a generated kernel's), on the calls a plan
    serves after it, for a transposed dense operand and for a vector.  Each is written in
    place, and the next call's product is unchanged by it."""
    rows, cols, free = SHAPES[shape]
    make, _ = OPERANDS[kind]
    d = dense_matrix(rows, cols, density=0.8, dtype=dtype)
    a = make(d)
    left = side == "left"
    inner = cols if left else rows
    if layout == "vector":
        dense = rand(inner, dtype=dtype) - 0.5
    else:
        shaped = (inner, free) if left else (free, inner)
        raw = rand(*(shaped[::-1] if layout == "transposed" else shaped), dtype=dtype)
        dense = (raw - 0.5).t() if layout == "transposed" else raw - 0.5
    ref = d.double() @ dense.double() if left else dense.double() @ d.double()
    with torch.no_grad():
        for _ in range(3):  # the first call, then calls a plan it installed serves
            out = scorch.matmul(a, dense) if left else scorch.matmul(dense, a)
            assert type(out) is torch.Tensor and out._base is None
            torch.testing.assert_close(out.double(), ref, **tolerance(dtype))
            out.add_(1.0)


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_parameter_operand_s_product_is_not_a_view(side):
    """A product with an nn.Parameter dense operand, twice, each checked at the Function and
    written in place after.  On the left the Function gets the Parameter, which is not a
    plain tensor, so no plan serves the product and each call takes the prebuilt kernel's
    route; on the right it gets the Parameter's transpose, a plain tensor."""
    d = dense_matrix()
    a = d.to_sparse_csr()
    left = side == "left"
    w = torch.nn.Parameter(rand(*((5, 3) if left else (4, 7))) - 0.5)
    with _recorded_by("python"), _function_outputs() as seen:
        outs = [scorch.matmul(a, w) if left else scorch.matmul(w, a) for _ in range(2)]
    _owned(seen, ["_SparseMM" if left else "_SparseXA"])
    ref = d @ w.detach() if left else w.detach() @ d
    for out in outs:
        torch.testing.assert_close(out.detach(), ref)
        out.relu_()


@pytest.mark.parametrize("recorded", [True, False], ids=["recorded", "unrecorded"])
def test_the_prebuilt_route_s_result_is_its_own_buffer(recorded, monkeypatch):
    """The prebuilt kernel allocates its product as the [rows, k] matrix it is (csrc/spmm.h),
    and the prebuilt route returns that tensor: the result is the kernel's own buffer, not a
    copy and not a view.  Nothing else holds it: after the call the result is its buffer's
    one handle, two calls with the same operands give two buffers, and writing the first
    result in place changes neither the second nor a third call's.  An nn.Parameter dense
    operand is not a plain tensor, so no plan serves the product and every call takes that
    route: in _SparseMM's forward (recorded) or in an ordinary call; the spy on the kernel's
    call fails the test where a call does not."""
    from scorch import ops

    taken = []
    execute = ops.execute_prebuilt_binary_kernel

    def spy(*args, **kwargs):
        result_cpp, shape = execute(*args, **kwargs)
        value = result_cpp.storage.value
        taken.append((value.data_ptr(), tuple(value.shape)))
        del value
        return result_cpp, shape

    monkeypatch.setattr(ops, "execute_prebuilt_binary_kernel", spy)
    d = dense_matrix()
    a = d.to_sparse_csr() if recorded else STensor.from_csr(d.to_sparse_csr())
    w = torch.nn.Parameter(rand(5, 3) - 0.5, requires_grad=recorded)
    ref = d @ w.detach()
    with _recorded_by("python"):
        first, second = [scorch.matmul(a, w) for _ in range(2)]
    assert taken == [(first.data_ptr(), (7, 3)), (second.data_ptr(), (7, 3))]
    for out in (first, second):
        assert out._base is None and out._use_count() == 1
        assert out.shape == (7, 3) and out.is_contiguous()
        assert (out.grad_fn is not None) == recorded
    assert first.data_ptr() != second.data_ptr()
    with torch.no_grad():
        first.add_(1.0)
    with _recorded_by("python"):
        third = scorch.matmul(a, w)
    assert len(taken) == 3 and taken[2] == (third.data_ptr(), (7, 3))
    torch.testing.assert_close(first.detach(), ref + 1.0)
    torch.testing.assert_close(second.detach(), ref)
    torch.testing.assert_close(third.detach(), ref)
    if recorded:
        (g,) = torch.autograd.grad(second.sum(), (w,))
        torch.testing.assert_close(g, d.t() @ torch.ones(7, 3, dtype=d.dtype))


def test_a_result_einsums_derivative_cannot_read_is_refused_by_name():
    """A product recorded as its einsum whose result the einsum's derivative cannot read
    (a torch sparse tensor) carries matmul_derivative_unsupported, and the message names
    the product, why it has no derivative, and which products have one
    (autograd._MATMUL_DIFFERENTIABLE).  No product scorch.matmul computes here returns
    such a result, so autograd.matmul is given an ``ordinary`` that does: the torch CSR
    product a torch CSR times torch CSR call returns once torch CSR results are kept."""
    d = dense_matrix(5, 5)
    a = d.to_sparse_csr().requires_grad_()
    b = d.t().contiguous().to_sparse_csr()

    def ordinary(x, y):
        return (x.detach().to_dense() @ y.to_dense()).to_sparse_csr()

    out = grad_mod.matmul(a, b, ordinary, {})
    assert type(out) is torch.Tensor and out.layout is torch.sparse_csr
    torch.testing.assert_close(out.detach().to_dense(), d @ d.t())
    assert out.grad_fn is not None
    with pytest.raises(NotImplementedError) as refused:
        out.to_dense().sum().backward()
    assert refused.value.code == "matmul_derivative_unsupported"
    what = "a 2-D torch.sparse_csr torch.float64 tensor"
    message = str(refused.value)
    assert message.startswith(
        f"scorch.matmul has no derivative for {what} @ {what} (the call is not in the "
        "form the derivative reads); differentiable products are "
    ), message
    assert message.endswith("as the einsum it is (scorch.einsum's derivative)"), message
    assert "float32/float64 CPU" in message and "scorch's SpMM" in message, message
    assert message.endswith("; " + grad_mod._MATMUL_DIFFERENTIABLE), message


def test_the_uncovered_products_function_returns_no_input():
    """A product the SpMM Functions do not cover is recorded by einsum_grad's _EinsumValues
    (a generated kernel's product, here), and a result with no derivative by
    _NotDifferentiable (not_differentiable).  Each returns ``.detach()`` of the result it
    is given: a new tensor, which costs less than returning the result itself (torch then
    makes a view of it).  The caller's leaf keeps its version and stays a leaf."""
    d = dense_matrix()
    a = d.to_sparse_csr()
    w = (rand(5, 3) - 0.5).requires_grad_()
    version = w._version
    with _function_outputs() as seen:
        out = scorch.matmul(a, w, use_cache=False)
        refused = grad_mod.not_differentiable(
            d @ w.detach(), "no derivative", "matmul_derivative_unsupported", (w,)
        )
    _owned(seen, ["_EinsumValues", "_NotDifferentiable"])
    assert w._version == version and w.grad_fn is None and w.is_leaf
    for y in (out, refused):
        y.relu_()
        torch.testing.assert_close(y.detach(), (d @ w.detach()).relu())


def test_sparse_times_sparse_is_unchanged():
    d = dense_matrix(5, 5)
    a = d.to_sparse_csr().requires_grad_()
    b = d.t().contiguous().to_sparse_csr()
    out = scorch.matmul(a, b)
    assert type(out) is STensor  # no autograd for an STensor result


# --------------------------------------------------------------------------- #
# The kernels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "rows,cols,density,k",
    [(9, 6, 0.4, 5), (5, 8, 0.0, 3), (1, 1, 1.0, 1), (40, 30, 0.2, 33)],
)
def test_sddmm_and_transpose_kernels(index_dtype, dtype, rows, cols, density, k):
    d = dense_matrix(rows, cols, density, dtype=dtype)
    csr = d.to_sparse_csr()
    crow = csr.crow_indices().to(index_dtype)
    col = csr.col_indices().to(index_dtype)
    x = rand(rows, k, dtype=dtype)
    y = rand(cols, k, dtype=dtype, seed=5)
    row_of = torch.repeat_interleave(torch.arange(rows), crow.diff().long())
    expected = (x[row_of] * y[col.long()]).sum(1)
    torch.testing.assert_close(
        scorch_ops.sddmm_csr(crow, col, x, y), expected, **tolerance(dtype)
    )

    t_crow, t_col, perm = scorch_ops.csr_transpose(crow, col, cols)
    assert t_crow.dtype == torch.int32
    t = torch.sparse_csr_tensor(
        t_crow.long(), t_col.long(), csr.values()[perm], (cols, rows)
    )
    assert torch.equal(t.to_dense(), d.t())
    for r in range(cols):  # each row of the transpose is sorted
        seg = t_col[t_crow[r] : t_crow[r + 1]]
        assert torch.all(seg[1:] > seg[:-1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [0, 1, 1000, 400_000], ids=lambda n: f"n{n}")
@pytest.mark.parametrize("nthreads", [-1, 4])
def test_gather_values_kernel(dtype, n, nthreads):
    """out[q] = values[perm[q]], serially and (from 393216 elements, with threads) split
    across torch's pool."""
    values = torch.rand(n, dtype=dtype)
    perm = torch.randperm(n)
    out = torch.empty_like(values)
    scorch_ops.gather_values(values, perm, out, nthreads)
    assert torch.equal(out, values[perm])


# --------------------------------------------------------------------------- #
# A transpose's values, gathered by the product's team where it has one
# --------------------------------------------------------------------------- #


def _wide_csr(dtype):
    """3000 x 60000 with rows of 0-60 nonzeros and three rows widened past 1/64 of the
    nonzeros, so a plan computes them in segments (test_spmm_wide_rows.py's matrix)."""
    g = torch.Generator().manual_seed(0)
    rows, cols = 3000, 60000
    lengths = torch.randint(0, 61, (rows,), generator=g)
    lengths[0], lengths[1500], lengths[2999] = 40000, 30000, 12000
    crow = torch.zeros(rows + 1, dtype=torch.long)
    crow[1:] = torch.cumsum(lengths, 0)
    col = torch.cat(
        [torch.randperm(cols, generator=g)[: int(n)].sort().values for n in lengths]
    )
    vals = torch.randn(int(crow[-1]), generator=g, dtype=dtype)
    return torch.sparse_csr_tensor(crow, col, vals, size=(rows, cols))


def _plan(a, kind, width, panels=(0, 0)):
    rows, cols = a.shape
    indices = [[], [a.crow_indices().to(torch.int32), a.col_indices().to(torch.int32)]]
    return scorch_ops.make_spmm_csr_plan(
        kind, [rows, cols], indices, a.values(), width, *panels
    )


GATHERED_CASES = [
    ("v2", torch.float32, (0, 0)),
    ("v2_double", torch.float64, (0, 0)),
    ("tilej", torch.float32, (0, 64)),
    ("tileijk", torch.float32, (8, 32)),
    ("reference", torch.float64, (0, 0)),
]


@pytest.mark.parametrize("kind,dtype,panels", GATHERED_CASES, ids=lambda c: str(c))
@pytest.mark.parametrize("width", [1, 2, 3, 4, 8, 16, 33, 64])
@pytest.mark.parametrize(
    "nthreads,atparallel", [(-1, False), (1, False), (4, False), (4, True)]
)
@pytest.mark.parametrize("wide", [False, True], ids=["plain", "wide_rows"])
def test_gathered_run_is_run_on_the_gathered_values(
    kind, dtype, panels, width, nthreads, atparallel, wide
):
    """run_gathered(buffer, src, perm) is run(src[perm]) bit for bit, and leaves the buffer
    holding src[perm]: on every v2 row kernel (widths 1-64 cover the exact, half-vector,
    register-block and register-tile loops), with and without wide rows in segments, on
    one worker and on a team, on torch's pool and on a private one; and on the kernels
    that take the buffer filled beforehand (tile-j, tile-ijk, reference)."""
    if wide and kind not in ("v2", "v2_double"):
        pytest.skip("wide rows are the v2 kernels'")
    if wide:
        a = _wide_csr(dtype)
    else:
        a = dense_matrix(300, 500, 0.05, dtype=dtype).to_sparse_csr()
    plan = _plan(a, kind, width, panels)
    assert plan is not None and plan.kind == kind
    if wide and kind.startswith("v2"):
        assert plan.wide_segments > 0
    nnz = a._nnz()
    g = torch.Generator().manual_seed(width)
    src = torch.randn(nnz, generator=g, dtype=dtype)
    perm = torch.randperm(nnz, generator=g)
    b = torch.randn(a.shape[1], width, generator=g, dtype=dtype)
    want = plan.run(src[perm], b, nthreads, atparallel)
    buffer = torch.full((nnz,), float("nan"), dtype=dtype)
    fills = plan.team_fills
    got = plan.run_gathered(buffer, src, perm, b, nthreads, atparallel)
    assert torch.equal(buffer, src[perm])
    assert torch.equal(got, want)
    # The team filled the values exactly when the product has one. Where the count depends
    # on whether the call runs on torch's full-width team, which the test cannot see, the
    # fill is left unasserted.
    team = {_workers(plan, width, nthreads, ft) > 1 for ft in (False, True)}
    if not kind.startswith("v2"):
        assert plan.team_fills == fills
    elif len(team) == 1:
        assert plan.team_fills == fills + (1 if team.pop() else 0)


def _workers(plan, width, nthreads, full_team):
    return scorch_ops.scorch_spmm_v2_workers(
        plan.nnz, plan.empty_rows, width, width, plan.rows, nthreads, full_team
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gathered_cases_run_both_fills(dtype):
    """run_gathered fills on the product's team where a v2 product has one and before the
    call where it runs on one worker (scorch_spmm_v2_workers decides). The grid of
    test_gathered_run_is_run_on_the_gathered_values holds both fills to `run` bit for bit
    only if it reaches both: here, whichever team width torch's pool resolves to, it has
    calls of each."""
    kind = "v2" if dtype == torch.float32 else "v2_double"
    matrices = [dense_matrix(300, 500, 0.05, dtype=dtype).to_sparse_csr(), _wide_csr(dtype)]
    for full_team in (False, True):
        counts = [
            _workers(_plan(a, kind, width), width, nthreads, full_team)
            for a in matrices
            for width in [1, 2, 3, 4, 8, 16, 33, 64]
            for nthreads in (-1, 1, 4)
        ]
        assert min(counts) == 1 and max(counts) > 1, (full_team, counts)


@pytest.mark.parametrize("kind,dtype,panels", GATHERED_CASES, ids=lambda c: str(c))
def test_plan_counts_its_empty_rows(kind, dtype, panels):
    """A v2 plan's empty rows are A's (what scorch_spmm_v2_workers credits); the other kinds
    never ask, and hold none."""
    a = dense_matrix(300, 500, 0.002, dtype=dtype).to_sparse_csr()
    empty = int((a.crow_indices().diff() == 0).sum())
    assert empty > 0
    plan = _plan(a, kind, 8, panels)
    assert plan.empty_rows == (empty if kind.startswith("v2") else 0)


def test_gathered_run_declines_what_it_cannot_serve():
    a = dense_matrix(40, 30, 0.2, dtype=torch.float32).to_sparse_csr()
    plan = _plan(a, "v2", 8)
    nnz = a._nnz()
    src, perm, b = torch.rand(nnz), torch.randperm(nnz), torch.rand(30, 8)
    assert plan.run_gathered(torch.empty(nnz), src, perm.int(), b) is None
    assert plan.run_gathered(torch.empty(nnz), src[1:], perm, b) is None
    assert plan.run_gathered(torch.empty(nnz), src.double(), perm, b) is None
    assert plan.run_gathered(src, src, perm, b) is None  # the buffer is src
    assert plan.run_gathered(torch.empty(nnz), src, perm, torch.rand(30, 9)) is None
    assert plan.run_gathered(torch.empty(nnz), src, perm, b) is not None


@pytest.mark.parametrize("kind,dtype,panels", GATHERED_CASES, ids=lambda c: str(c))
def test_gathered_run_declines_what_run_declines(kind, dtype, panels):
    """run_gathered checks its buffer and `b` with a copy of run's checks (SpmmCsrPlan::accepts),
    so that run's own code is unchanged; this holds the two to the same calls, each operand
    perturbed one way at a time, and the structure changed under the plan."""
    a = dense_matrix(40, 30, 0.2, dtype=dtype).to_sparse_csr()
    pos, crd = a.crow_indices().to(torch.int32), a.col_indices().to(torch.int32)
    plan = scorch_ops.make_spmm_csr_plan(kind, [40, 30], [[], [pos, crd]], a.values(), 8, *panels)
    assert plan is not None and plan.kind == kind
    nnz = a._nnz()
    other = torch.float64 if dtype == torch.float32 else torch.float32
    src, perm = torch.rand(nnz, dtype=dtype), torch.randperm(nnz)
    v, b = torch.rand(nnz, dtype=dtype), torch.rand(30, 8, dtype=dtype)
    calls = {
        "served": (v, b, -1),
        "served, 4 threads": (v, b, 4),
        "values dtype": (v.to(other), b, -1),
        "b dtype": (v, b.to(other), -1),
        "values length": (torch.rand(nnz + 1, dtype=dtype), b, -1),
        "values 2-D": (v.view(1, nnz), b, -1),
        "b rows": (v, torch.rand(31, 8, dtype=dtype), -1),
        "b width": (v, torch.rand(30, 9, dtype=dtype), -1),
        "b 1-D": (v, torch.rand(30, dtype=dtype), -1),
        "values strided": (torch.rand(2 * nnz, dtype=dtype)[::2], b, -1),
        "b transposed": (v, torch.rand(8, 30, dtype=dtype).t(), -1),
        "values negative view": (torch._neg_view(v), b, -1),
        "b negative view": (v, torch._neg_view(b), -1),
        "b sparse": (v, b.to_sparse(), -1),
        "nthreads -2": (v, b, -2),
        "nthreads past int": (v, b, 2**31),
    }
    for name, (values, operand, nthreads) in calls.items():
        ran = plan.run(values, operand, nthreads) is not None
        # `values` is the buffer: run_gathered checks it where run checks A's values
        gathered = plan.run_gathered(values, src, perm, operand, nthreads) is not None
        assert ran == gathered, name
        assert ran == name.startswith("served"), name
    pos.add_(0)  # a write moves the version counter: the structure is no longer the validated one
    assert plan.run(v, b) is None
    assert plan.run_gathered(torch.empty(nnz, dtype=dtype), src, perm, b) is None


@pytest.mark.parametrize("kind", ["csr", "stensor"])
def test_the_planned_transpose_gathers_values_written_through_data(kind, monkeypatch):
    """x @ A once A's transpose carries a plan: the plan gathers A's values on every call
    (planned_matmul_gathered serves it, and for a torch CSR operand, once that has recorded
    itself, the C entry), so a write through .data -- which moves no version counter --
    reaches the next product."""
    served = []
    real = scorch_ops.planned_matmul_gathered

    def spy(*args):
        out = real(*args)
        served.append(out is not None)
        return out

    monkeypatch.setattr(scorch_ops, "planned_matmul_gathered", spy)
    a = dense_matrix(200, 150, 0.05, dtype=torch.float32).to_sparse_csr()
    x = rand(4, 200, dtype=torch.float32)
    operand = a if kind == "csr" else STensor.from_torch(a)
    values = a.values() if kind == "csr" else operand._storage._value
    for step in range(6):
        values.data.mul_(1.5)
        got = scorch.matmul(x, operand)
        want = (
            x
            @ torch.sparse_csr_tensor(
                a.crow_indices(), a.col_indices(), values.detach().clone(), a.shape
            ).to_dense()
        )
        torch.testing.assert_close(got, want, **tolerance(torch.float32))
    if kind == "csr":
        # Sighted, planned, served by the carrier's plan (which records itself), then
        # served by the C entry from that record.
        assert served == [False, False, True] and _served(a) == 3
    else:
        assert served[-3:] == [True, True, True]
