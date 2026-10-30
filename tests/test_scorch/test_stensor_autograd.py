"""Autograd through an STensor's stored values (drop-in note, section 0.2).

Every gradient here is checked two ways: ``torch.autograd.gradcheck`` (and
``gradgradcheck`` where the note asks for a second derivative) in float64, which
compares the recorded derivative with finite differences, and a comparison of the
values and gradients with the same computation written in dense torch, in float64.
The invariants of section 0.2 are asserted after the calls that could break them.
"""

from __future__ import annotations

import copy
import pathlib
import pickle
import re
import sys
import types

import pytest
import torch
import torch.autograd.forward_ad as fwAD

import scorch
from scorch import autograd as scorch_autograd
from scorch import stensor as stensor_module
from scorch import stensor_grad
from scorch.exceptions import (
    EinsumDerivativeError,
    MatmulDerivativeError,
    ScorchError,
    SparseLinearDerivativeError,
)
from scorch.plan import GRAD_ATTR
from scorch.stensor import STensor
from tests.test_scorch.kernel_cache_support import empty_kernel_caches, kernel_caches

F64 = torch.float64
AUTOGRAD_VALUES = stensor_grad.AUTOGRAD_VALUES


# -- helpers ------------------------------------------------------------------------


def _dense(shape, density=0.5, seed=0, dtype=F64):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=dtype) + 0.5
    mask = torch.rand(shape, generator=generator) < density
    return values * mask


def _layout(dense, fmt, mode_order=None):
    """An STensor of ``dense`` stored in ``fmt`` (and ``mode_order``), without history:
    the structure the tests build tensors over."""

    with torch.no_grad():
        tensor = scorch.from_torch(dense.detach().clone())
        if mode_order is not None:
            tensor.change_mode_order(list(mode_order))
        if any(level != "d" for level in fmt):
            tensor.to_sparse(fmt)
    return tensor


def _over(template, values):
    """An STensor over ``template``'s structure whose values are ``values``, built by the
    public constructor, so its history links to ``values``."""

    return STensor.from_components(
        tuple(template.shape),
        template.format,
        [list(arrays) for arrays in template._native_mode_indices()],
        values,
        mode_order=list(template.mode_order),
    )


def _logical(template, values):
    """The dense torch tensor whose stored entries are ``values`` at ``template``'s
    coordinates: the reference, differentiable in ``values``."""

    keys = stensor_grad._Structure.of(template).keys()
    shape = tuple(template.logical_shape)
    flat = torch.zeros(int(torch.tensor(shape).prod()), dtype=values.dtype)
    return flat.index_add(0, keys, values).reshape(shape)


def _start(template):
    return template._raw_values.clone().requires_grad_()


def _check(tensor):
    """Section 0.2's per-tensor invariants."""

    assert not tensor._storage._value.requires_grad
    tracked = tensor.__dict__.get(AUTOGRAD_VALUES)
    assert tensor._metadata.requires_grad == (tracked is not None)
    if tracked is not None:
        assert tracked.requires_grad
        assert tracked.dim() == 1 and tracked.is_contiguous()
        assert tracked.data_ptr() == tensor._storage._value.data_ptr()
        assert tracked.numel() == tensor._storage._value.numel()


def _dense_of(result):
    if type(result) is STensor:
        _check(result)
        return result.to_torch(in_place=False)
    return result


def _close(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


# -- what reaches native code (scorch-merge, 13:4x) ---------------------------------

# The scorch_ops entries that read a torch sparse tensor's structure and never its
# values: they are handed the caller's own tensor, which may require grad.
_STRUCTURE_READERS = {
    "torch_sparse_index_members",
    "torch_structure_count",
    "torch_structure_state",
    "torch_structure_sweep",
}

# The functions outside the autograd modules (and this file) that read ``values`` of an
# STensor whose values autograd tracks.  Both hand it to torch, never to native code:
# sparse_linear_fm to its float64 fallback and to autograd.linear_fm, and scorch.compile
# to its check for negative, conjugate and forward-tangent inputs.
_VALUES_READERS = {
    ("scorch.ops", "sparse_linear_fm"),
    ("scorch.trace", "_prebuilt_inputs"),
}
_AUTOGRAD_MODULES = {
    "scorch.stensor",
    "scorch.stensor_grad",
    "scorch.autograd",
    "scorch.einsum_grad",
}


def _plain_tensors(item, seen):
    """The torch tensors among a native call's arguments.  An STensor argument is not
    entered: native code reads it through its exact descriptors, which are detached."""

    if id(item) in seen:
        return
    seen.add(id(item))
    if isinstance(item, torch.Tensor):
        yield item
    elif isinstance(item, dict):
        for value in item.values():
            yield from _plain_tensors(value, seen)
    elif isinstance(item, (list, tuple)):
        for value in item:
            yield from _plain_tensors(value, seen)


@pytest.fixture(autouse=True)
def native_boundary(monkeypatch):
    """Every test in this file runs with every public callable of the scorch_ops
    extension and of every generated kernel module wrapped, and with STensor's
    ``values`` getter wrapped.  At the end of the test:

    * no native call was handed a tensor that requires grad while grad mode was on,
      where native code would drop its gradient or record history it cannot
      differentiate (inside a Function's forward, and in a backward that builds no
      graph, grad mode is off);
    * outside the autograd modules and this file, only the functions in
      ``_VALUES_READERS`` read ``values`` of an STensor whose values autograd tracks,
      grad mode on or off.  With the source test below, which allows the attribute's
      name only in the autograd modules, that is what keeps the tracked values out of
      native code.

    The LoopIR modules are covered the same way: their operands reach native code
    through ops' runtime snapshot, which reads ``_raw_values``.  Forward-mode tangents
    are not checked here: a Function's ``jvp`` hands native code a dual tensor's primal
    on purpose; the forward-mode tests compare the tangents themselves."""

    import scorch_ops

    from scorch import utils

    crossings, readers, wrapped = [], set(), set()

    def checked(where, fn):
        def call(*args, **kwargs):
            if (
                where not in _STRUCTURE_READERS
                and torch.is_grad_enabled()
                and any(
                    tensor.requires_grad
                    for tensor in _plain_tensors((args, kwargs), set())
                )
            ):
                caller = sys._getframe(1)
                crossings.append(
                    (where, caller.f_globals.get("__name__"), caller.f_code.co_name)
                )
            return fn(*args, **kwargs)

        return call

    def wrap(module, prefix):
        for name, fn in list(vars(module).items()):
            if callable(fn) and not isinstance(fn, type) and not name.startswith("_"):
                monkeypatch.setattr(module, name, checked(prefix + name, fn))

    def wrap_kernel(module):
        # Only a loaded module: a test elsewhere may leave a stand-in in utils._so_cache
        # (test_compile_options' fake builds return object()), and this process may
        # run that test before this file.
        if isinstance(module, types.ModuleType) and id(module) not in wrapped:
            wrapped.add(id(module))
            wrap(module, module.__name__ + ".")
        return module

    wrap(scorch_ops, "")
    for module in list(utils._so_cache.values()):
        wrap_kernel(module)
    for loader in ("_load_extension_file", "_build_and_load_extension"):
        original = getattr(utils, loader)
        monkeypatch.setattr(
            utils,
            loader,
            lambda *args, _load=original, **kwargs: wrap_kernel(_load(*args, **kwargs)),
        )
    # The caches that hold a native function itself, resolved before this test.
    empty_kernel_caches(monkeypatch)

    values_property = STensor.values

    def values(self):
        if AUTOGRAD_VALUES in self.__dict__:
            caller = sys._getframe(1)
            readers.add((caller.f_globals.get("__name__"), caller.f_code.co_name))
        return values_property.fget(self)

    monkeypatch.setattr(
        STensor,
        "values",
        property(values, values_property.fset, None, values_property.__doc__),
    )
    yield
    assert crossings == [], crossings
    outside = {
        reader
        for reader in readers
        if reader[0] not in _AUTOGRAD_MODULES and reader[0] != __name__
    }
    assert outside <= _VALUES_READERS, outside


FORMATS = [("ds", None), ("ds", (1, 0)), ("ss", None), ("oo", None)]


# -- products -----------------------------------------------------------------------


@pytest.mark.parametrize("fmt,order", FORMATS)
@pytest.mark.parametrize("side", ["left", "right"])
def test_matmul_of_a_tracked_stensor(fmt, order, side):
    template = _layout(_dense((5, 4), seed=1), fmt, order)
    v = _start(template)
    if side == "left":
        b = torch.rand(4, 3, dtype=F64, requires_grad=True)
    else:
        b = torch.rand(2, 5, dtype=F64, requires_grad=True)

    def scorch_f(v, b):
        a = _over(template, v)
        _check(a)
        out = scorch.matmul(a, b) if side == "left" else scorch.matmul(b, a)
        return _dense_of(out)

    def torch_f(v, b):
        a = _logical(template, v)
        return a @ b if side == "left" else b @ a

    _close(scorch_f(v, b), torch_f(v, b))
    assert torch.autograd.gradcheck(scorch_f, (v, b))
    assert torch.autograd.gradgradcheck(scorch_f, (v, b))


@pytest.mark.parametrize("fmt,order", FORMATS)
def test_matrix_vector_products(fmt, order):
    template = _layout(_dense((5, 4), seed=2), fmt, order)
    v = _start(template)
    x = torch.rand(4, dtype=F64, requires_grad=True)
    y = torch.rand(5, dtype=F64, requires_grad=True)

    def scorch_f(v, x, y):
        a = _over(template, v)
        return _dense_of(scorch.matmul(a, x)), _dense_of(scorch.matmul(y, a))

    assert torch.autograd.gradcheck(scorch_f, (v, x, y))


@pytest.mark.parametrize("fmt_a,fmt_b", [("ds", "ds"), ("ss", "ds"), ("ds", "dd")])
def test_matmul_of_two_stensors_is_the_einsum_derivative(fmt_a, fmt_b):
    ta = _layout(_dense((4, 5), seed=3), fmt_a)
    tb = _layout(_dense((5, 3), seed=4), fmt_b)
    va, vb = _start(ta), _start(tb)

    def scorch_f(va, vb):
        return _dense_of(scorch.matmul(_over(ta, va), _over(tb, vb)))

    _close(scorch_f(va, vb), _logical(ta, va) @ _logical(tb, vb))
    assert torch.autograd.gradcheck(scorch_f, (va, vb))


def test_matmul_with_a_format_keyword_records_the_same_result():
    template = _layout(_dense((5, 4), seed=5), "ds")
    v = _start(template)
    b = torch.rand(4, 3, dtype=F64, requires_grad=True)
    with torch.no_grad():
        untracked = scorch.matmul(_over(template, v.detach()), b.detach(), format="dd")

    def scorch_f(v, b):
        return _dense_of(scorch.matmul(_over(template, v), b, format="dd"))

    assert torch.equal(_dense_of(untracked), scorch_f(v, b).detach())
    assert torch.autograd.gradcheck(scorch_f, (v, b))


# -- einsum -------------------------------------------------------------------------

EINSUM_PROGRAMS = [
    # (expression, operands as (shape, format or None for a dense torch tensor), kwargs)
    ("ij,jk->ik", [((4, 5), "ds"), ((5, 3), None)], {"format": "dd"}),
    ("ij,jk->ik", [((4, 5), "ds"), ((5, 3), None)], {"format": "ds"}),
    ("ij,jk->ik", [((4, 5), "ds"), ((5, 3), "ds")], {}),
    ("ij,ij->ij", [((4, 5), "ds"), ((4, 5), None)], {}),
    ("ij,ik,jk->ij", [((4, 4), "oo"), ((4, 3), None), ((4, 3), None)], {}),
    ("ij->i", [((4, 5), "ds")], {"format": "d"}),
    ("ij,j->i", [((4, 5), "ss"), ((5,), None)], {"format": "d"}),
    ("ijk,k->ij", [((3, 4, 2), "dss"), ((2,), None)], {"format": "dd"}),
    ("i,i->", [((6,), "s"), ((6,), None)], {}),
]


def _einsum_case(case, seed=10):
    expression, specs, kwargs = case
    templates, leaves = [], []
    for k, (shape, fmt) in enumerate(specs):
        if fmt is None:
            templates.append(None)
            leaves.append(torch.rand(shape, dtype=F64, requires_grad=True))
        else:
            template = _layout(_dense(shape, density=0.6, seed=seed + k), fmt)
            templates.append(template)
            leaves.append(_start(template))
    return expression, templates, leaves, kwargs


def _einsum_operands(templates, leaves):
    return [
        leaf if template is None else _over(template, leaf)
        for template, leaf in zip(templates, leaves)
    ]


@pytest.mark.parametrize("case", EINSUM_PROGRAMS, ids=lambda case: case[0])
def test_einsum_through_the_compiler(case):
    expression, templates, leaves, kwargs = _einsum_case(case)

    def scorch_f(*leaves):
        return _dense_of(
            scorch.einsum(expression, *_einsum_operands(templates, leaves), **kwargs)
        )

    def torch_f(*leaves):
        operands = [
            leaf if template is None else _logical(template, leaf)
            for template, leaf in zip(templates, leaves)
        ]
        return torch.einsum(expression, *operands)

    _close(scorch_f(*leaves), torch_f(*leaves))
    assert torch.autograd.gradcheck(scorch_f, tuple(leaves))


@pytest.mark.parametrize(
    "case", [EINSUM_PROGRAMS[0], EINSUM_PROGRAMS[4]], ids=lambda case: case[0]
)
def test_einsum_second_derivative(case):
    expression, templates, leaves, kwargs = _einsum_case(case)

    def scorch_f(*leaves):
        return _dense_of(
            scorch.einsum(expression, *_einsum_operands(templates, leaves), **kwargs)
        )

    assert torch.autograd.gradgradcheck(scorch_f, tuple(leaves))


def test_einsum_with_explicit_zeros_and_an_empty_operand():
    template = _layout(_dense((4, 5), density=0.6, seed=20), "ds")
    v = _start(template)
    with torch.no_grad():
        v[::2] = 0.0  # explicit stored zeros: their gradient is still computed
    empty = _layout(torch.zeros(4, 5, dtype=F64), "ds")
    e = _start(empty)
    b = torch.rand(5, 3, dtype=F64, requires_grad=True)

    def scorch_f(v, e, b):
        a = _over(template, v)
        z = _over(empty, e)
        return (
            _dense_of(scorch.einsum("ij,jk->ik", a, b, format="dd")),
            _dense_of(scorch.einsum("ij,jk->ik", z, b, format="dd")),
        )

    assert torch.autograd.gradcheck(scorch_f, (v, e, b))
    out, _ = scorch_f(v, e, b)
    out.sum().backward()
    assert v.grad is not None and bool((v.grad[::2] != 0).any())


def test_einsum_forward_mode_into_a_dense_torch_result():
    template = _layout(_dense((4, 5), seed=21), "ds")
    a = _over(template, template._raw_values.clone())
    b = torch.rand(5, 3, dtype=F64)
    tangent = torch.rand(5, 3, dtype=F64)
    with fwAD.dual_level():
        dual = fwAD.make_dual(b, tangent)
        out = scorch.matmul(a, dual)
        got = fwAD.unpack_dual(out).tangent
    _close(got, _logical(template, a._raw_values) @ tangent)


def test_einsum_forward_tangent_into_an_stensor_without_history_is_refused():
    template = _layout(_dense((4, 5), seed=22), "ds")
    a = _over(template, template._raw_values.clone())
    b = torch.rand(5, 3, dtype=F64)
    with fwAD.dual_level():
        dual = fwAD.make_dual(b, torch.rand(5, 3, dtype=F64))
        with pytest.raises(ScorchError) as raised:
            scorch.einsum("ij,jk->ik", a, dual, format="ds")
    assert raised.value.code == "stensor_forward_tangent_unsupported"


def test_einsum_refuses_a_backward_after_its_index_arrays_were_written():
    template = _layout(_dense((4, 5), seed=23), "ds")
    v = _start(template)
    a = _over(template, v)
    b = torch.rand(5, 3, dtype=F64, requires_grad=True)
    out = scorch.einsum("ij,jk->ik", a, b, format="ds")
    a._native_mode_indices()[1][1].add_(0)  # moves the version counter, not the values
    with pytest.raises(ScorchError) as raised:
        out.to_torch(in_place=False).sum().backward()
    assert raised.value.code == "stensor_structure_written_before_backward"


# -- conversions and addition -------------------------------------------------------

# (format, mode order, density, conversion). ``to_sparse`` stores only the nonzeros, so
# its derivative at a zero is zero, as torch's ``to_sparse`` has it, while a finite
# difference there is one: its source holds no zeros.
CONVERSIONS = [
    ("ds", None, 0.5, lambda t: t.to_dense(in_place=False)),
    ("ds", None, 0.5, lambda t: t.to_dense(in_place=True)),
    ("dd", None, 1.0, lambda t: t.to_sparse("ds")),
    ("ds", None, 0.5, lambda t: t.change_mode_order([1, 0])),
    ("oo", None, 0.5, lambda t: t.change_mode_order([1, 0])),
    ("ds", (1, 0), 0.5, lambda t: t.to_dense(in_place=False)),
    ("ss", None, 0.5, lambda t: t.copy()),
    ("ds", None, 0.5, lambda t: t.coalesce()),
]


@pytest.mark.parametrize("fmt,order,density,convert", CONVERSIONS)
def test_conversions(fmt, order, density, convert):
    template = _layout(_dense((4, 5), density=density, seed=30), fmt, order)
    v = _start(template)

    def scorch_f(v):
        source = _over(template, v)
        result = convert(source)
        _check(source)
        return _dense_of(result)

    _close(scorch_f(v), _logical(template, v))
    assert torch.autograd.gradcheck(scorch_f, (v,))
    assert torch.autograd.gradgradcheck(scorch_f, (v,))


@pytest.mark.parametrize("fmt_a,fmt_b", [("ds", "ds"), ("ds", "dd"), ("oo", "ds")])
def test_addition(fmt_a, fmt_b):
    ta = _layout(_dense((4, 5), seed=31), fmt_a)
    tb = _layout(_dense((4, 5), seed=32), fmt_b)
    va, vb = _start(ta), _start(tb)

    def scorch_f(va, vb):
        return _dense_of(_over(ta, va) + _over(tb, vb))

    _close(scorch_f(va, vb), _logical(ta, va) + _logical(tb, vb))
    assert torch.autograd.gradcheck(scorch_f, (va, vb))


def test_a_chain_of_product_conversion_and_a_second_product():
    template = _layout(_dense((5, 4), seed=33), "ds")
    v = _start(template)
    b = torch.rand(4, 3, dtype=F64, requires_grad=True)
    c = torch.rand(5, 2, dtype=F64, requires_grad=True)

    def scorch_f(v, b, c):
        a = _over(template, v)
        product = scorch.einsum("ij,jk->ik", a, b, format="ds")
        # A relayout: the logical tensor, and so its labels, stay as they were.
        product.change_mode_order([1, 0])
        second = scorch.einsum("ik,ij->kj", product, c, format="dd")
        return _dense_of(second)

    def torch_f(v, b, c):
        return (_logical(template, v) @ b).T @ c

    _close(scorch_f(v, b, c), torch_f(v, b, c))
    assert torch.autograd.gradcheck(scorch_f, (v, b, c))


# -- in-place conversion of a leaf and of a non-leaf --------------------------------


def test_in_place_conversion_keeps_the_leaf_and_a_graph_built_before_it():
    dense = _dense((5, 4), seed=40)
    a = scorch.from_torch(dense.clone())
    a.requires_grad = True
    b = torch.rand(4, 3, dtype=F64)
    before = scorch.matmul(a, b)  # recorded on the dense layout
    a.to_sparse("ds")
    _check(a)
    assert a.is_leaf and a.grad_fn is None
    after = scorch.matmul(a, b)
    (before.sum() + after.sum()).backward()
    expected = torch.ones(5, 3, dtype=F64) @ b.T  # d(sum(A @ b))/dA, twice
    stored = stensor_grad._Structure.of(a).keys()
    got = a.grad
    assert got is not None and got.format == a.format
    _close(got.values, 2 * expected.reshape(-1)[stored])


def test_in_place_conversion_of_a_non_leaf():
    template = _layout(_dense((5, 4), seed=41), "ds")
    v = _start(template)
    b = torch.rand(4, 3, dtype=F64)
    a = _over(template, v) + _over(template, v)  # a non-leaf
    assert not a.is_leaf
    a.change_mode_order([1, 0])
    _check(a)
    scorch.matmul(a, b).sum().backward()
    expected = (torch.ones(5, 3, dtype=F64) @ b.T) * 2
    keys = stensor_grad._Structure.of(template).keys()
    _close(v.grad, expected.reshape(-1)[keys])


@pytest.mark.parametrize(
    "shape,fmt,mode_order",
    [
        ((5, 4), "ds", (1, 0)),
        ((5, 4), "ss", (1, 0)),
        ((3, 4, 5), "dss", (2, 0, 1)),
        ((3, 4, 5), "sds", (1, 2, 0)),
    ],
)
def test_structure_keys_are_row_major_positions_in_storage_order(
    shape, fmt, mode_order
):
    """``_Structure.keys()`` is the row-major position in the logical shape of each
    stored value, in storage order.  The tests above use it as their reference; here
    it is checked against positions read off distinct values, on layouts whose level
    order is not the logical one."""

    positions = torch.arange(1, torch.Size(shape).numel() + 1, dtype=F64)
    dense = positions.reshape(shape) * (_dense(shape, seed=71) != 0)
    template = _layout(dense, fmt, mode_order)
    permutation = list(template._storage.layout.permutation)
    assert permutation != sorted(permutation)
    stored = template._raw_values
    assert bool((stored != 0).all())  # every stored value is a position
    keys = stensor_grad._Structure.of(template).keys()
    assert torch.equal(keys, (stored - 1).to(torch.int64))


# -- constructors from torch tensors that require grad ------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda w: scorch.from_torch(w).to_sparse("ds"),
        lambda w: scorch.from_csr(w.to_sparse_csr(), "w"),
        lambda w: scorch.from_coo(w.to_sparse_coo(), name="w"),
        lambda w: scorch.from_torch(w.to_sparse_csr()),
        lambda w: scorch.from_torch(w.to_sparse_coo()),
    ],
    ids=[
        "from_torch_dense",
        "from_csr",
        "from_coo",
        "from_torch_csr",
        "from_torch_coo",
    ],
)
def test_constructors_carry_a_parameters_gradient(build):
    dense = _dense((5, 4), seed=50)
    w = torch.nn.Parameter(dense.clone())
    x = torch.rand(4, 3, dtype=F64)
    a = build(w)
    _check(a)
    scorch.matmul(a, x).pow(2).sum().backward()
    reference = torch.nn.Parameter(dense.clone())
    mask = (dense != 0).to(F64)
    ((reference * mask) @ x).pow(2).sum().backward()
    _close(w.grad, reference.grad * mask)


# -- sparse_linear_fm, sparse_linear, fast_transpose, scorch.compile ----------------


@pytest.mark.parametrize("activation", [None, "relu", "sigmoid"])
def test_sparse_linear_fm_float32_kernel(activation):
    dense = _dense((6, 5), seed=60, dtype=torch.float32)
    weight = scorch.from_csr(dense.to_sparse_csr(), "w")
    weight.requires_grad = True
    x = torch.rand(5, 4, dtype=torch.float32, requires_grad=True)
    bias = torch.rand(6, dtype=torch.float32, requires_grad=True)
    out = scorch.sparse_linear_fm(x, weight, bias, activation)
    _check(weight)
    with torch.no_grad():
        untracked = scorch.sparse_linear_fm(x, weight, bias, activation)
    assert torch.equal(out.detach(), untracked)

    w64 = dense.to(F64).requires_grad_()
    x64 = x.detach().to(F64).requires_grad_()
    b64 = bias.detach().to(F64).requires_grad_()
    z = w64 @ x64 + b64[:, None]
    ref = {None: z, "relu": torch.relu(z), "sigmoid": torch.sigmoid(z)}[activation]
    g = torch.rand_like(out)
    out.backward(g)
    ref.backward(g.to(F64))
    keys = stensor_grad._Structure.of(weight).keys()
    torch.testing.assert_close(
        weight.grad.values.to(F64), w64.grad.reshape(-1)[keys], rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(x.grad.to(F64), x64.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(bias.grad.to(F64), b64.grad, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("activation", [None, "sigmoid"])
def test_sparse_linear_fm_float64_gradcheck(activation):
    template = _layout(_dense((6, 5), seed=61), "ds")
    v = _start(template)
    x = torch.rand(5, 4, dtype=F64, requires_grad=True)
    bias = torch.rand(6, dtype=F64, requires_grad=True)

    def scorch_f(v, x, bias):
        return scorch.sparse_linear_fm(x, _over(template, v), bias, activation)

    # In float64 throughout: the bias was once rounded to float32 here (probe 84014).
    z = _logical(template, v) @ x + bias[:, None]
    _close(scorch_f(v, x, bias), torch.sigmoid(z) if activation else z)
    assert torch.autograd.gradcheck(scorch_f, (v, x, bias))


def test_sparse_linear_with_a_dense_parameter_weight():
    dense = _dense((6, 5), seed=62, dtype=torch.float32)
    w = torch.nn.Parameter(dense.clone())
    x = torch.rand(4, 5, dtype=torch.float32, requires_grad=True)
    out = scorch.sparse_linear(x, w, None, "relu")
    out.sum().backward()
    w64 = dense.to(F64).requires_grad_()
    x64 = x.detach().to(F64).requires_grad_()
    torch.relu(x64 @ (w64 * (dense != 0).to(F64)).T).sum().backward()
    torch.testing.assert_close(x.grad.to(F64), x64.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        w.grad.to(F64), w64.grad * (dense != 0).to(F64), rtol=1e-4, atol=1e-5
    )


def test_sparse_linear_fm_refuses_the_derivative_of_a_csc_weight():
    template = _layout(_dense((6, 5), seed=63, dtype=torch.float32), "ds", (1, 0))
    weight = _over(template, template._raw_values.clone())
    weight.requires_grad = True
    x = torch.rand(6, 4, dtype=torch.float32)
    with torch.no_grad():
        untracked = scorch.sparse_linear_fm(x, weight)
    out = scorch.sparse_linear_fm(x, weight)
    assert torch.equal(out.detach(), untracked)
    with pytest.raises(ScorchError) as raised:
        out.sum().backward()
    assert raised.value.code == "sparse_linear_derivative_unsupported"
    assert type(raised.value) is SparseLinearDerivativeError


@pytest.mark.parametrize(
    "code,error",
    [
        ("matmul_derivative_unsupported", MatmulDerivativeError),
        ("einsum_derivative_unsupported", EinsumDerivativeError),
        ("sparse_linear_derivative_unsupported", SparseLinearDerivativeError),
    ],
)
def test_a_result_with_no_derivative_raises_its_operations_class(code, error):
    """Each code a result with no derivative is recorded with raises its operation's
    class, which carries the code on the class: in backward mode and in forward mode,
    and a NotImplementedError either way, as torch raises for an operation with none."""

    x = torch.rand(3, dtype=F64, requires_grad=True)
    out = scorch_autograd.not_differentiable(
        x.detach() * 2, "no derivative", code, (x,)
    )
    assert out.requires_grad
    with pytest.raises(error) as raised:
        out.sum().backward()
    assert raised.value.code == code
    assert isinstance(raised.value, NotImplementedError)
    assert str(raised.value) == "no derivative"

    with fwAD.dual_level():
        dual = fwAD.make_dual(x.detach(), torch.ones(3, dtype=F64))
        with pytest.raises(error) as raised:
            scorch_autograd.not_differentiable(
                dual.detach() * 2, "no derivative", code, (dual,)
            )
    assert raised.value.code == code
    assert str(raised.value) == "(forward mode) no derivative"


def test_fast_transpose_records_its_derivative():
    x = torch.rand(7, 5, dtype=torch.float32, requires_grad=True)
    y = scorch.fast_transpose(x)
    assert torch.equal(y.detach(), x.detach().T.contiguous())
    g = torch.rand(5, 7, dtype=torch.float32)
    y.backward(g)
    assert torch.equal(x.grad, g.T)
    x64 = torch.rand(4, 3, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(scorch.fast_transpose, (x64,))


# -- outputs that take in-place writes (the note above autograd._in_layout) ----------
#
# Each Function the unit adds returns a tensor that is not a view, so what it returns (an
# STensor's values, or the tensor sparse_linear_fm and fast_transpose return) takes an
# in-place write as torch's own results do: relu_ in grad mode, as nn.ReLU(inplace=True)
# does, and add_ under no_grad, as an optimizer step does, after which its grad_fn is
# read.  The gradients are dense float64 torch's for the same write, first and second
# order.

WRITES = ["relu_ in grad mode", "add_ under no_grad"]


def _write(write, tensor):
    if write == "relu_ in grad mode":
        tensor.relu_()
    else:
        with torch.no_grad():
            tensor.add_(1.0)
        if tensor.requires_grad:
            assert tensor.grad_fn is not None


def _signed(template):
    """Leaf values of both signs at ``template``'s stored entries, for relu_ to change."""
    return (template._raw_values.clone() - 1.0).requires_grad_()


def _signed_dense(*shape, seed, dtype=F64):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=dtype) - 0.5
    return values.requires_grad_()


def _case_moved():
    template = _layout(_dense((4, 5), seed=70), "ds")
    return (
        "_MovedValuesBackward",
        [_signed(template)],
        lambda v: _over(template, v).change_mode_order([1, 0]),
        lambda v: _logical(template, v),
    )


def _case_summed():
    ta = _layout(_dense((4, 5), seed=71), "ds")
    tb = _layout(_dense((4, 5), seed=72), "ds")
    return (
        "_SummedValuesBackward",
        [_signed(ta), _signed(tb)],
        lambda va, vb: _over(ta, va) + _over(tb, vb),
        lambda va, vb: _logical(ta, va) + _logical(tb, vb),
    )


def _case_copied():
    # A transposed operand is not contiguous, so the STensor's values are a native copy.
    return (
        "_CopiedValuesBackward",
        [_signed_dense(5, 4, seed=73)],
        lambda w: STensor.from_torch(w.t()),
        lambda w: w.t(),
    )


def _case_einsum():
    template = _layout(_dense((4, 5), seed=74), "ds")
    return (
        "_EinsumValuesBackward",
        [_signed(template), _signed_dense(5, 3, seed=75)],
        lambda v, b: scorch.einsum("ij,jk->ik", _over(template, v), b, format="ds"),
        lambda v, b: _logical(template, v) @ b,
    )


def _case_linear_fm():
    # float32: the fused kernel's dtype (float64 takes torch's fallback, no Function).
    template = _layout(_dense((6, 5), seed=76, dtype=torch.float32), "ds")
    return (
        "_SparseLinearFMBackward",
        [
            _signed(template),
            _signed_dense(5, 4, seed=77, dtype=torch.float32),
            _signed_dense(6, seed=78, dtype=torch.float32),
        ],
        lambda v, x, bias: scorch.sparse_linear_fm(x, _over(template, v), bias, None),
        lambda v, x, bias: _logical(template, v) @ x + bias[:, None],
    )


def _case_fast_transpose():
    # float32: the native kernel is float32-only, and only it records
    # _FastTransposeBackward.  float64 takes torch's own copy; see
    # test_fast_transpose_of_float64_is_torchs_copy_and_takes_in_place_writes.
    return (
        "_FastTransposeBackward",
        [_signed_dense(7, 5, seed=79, dtype=torch.float32)],
        lambda x: scorch.fast_transpose(x),
        lambda x: x.t().contiguous(),
    )


# Each: (the node that records the output, the leaves, the computation, its dense
# float64 reference as a function of the same leaves).
WRITE_CASES = {
    "_MovedValues": _case_moved,
    "_SummedValues": _case_summed,
    "_CopiedValues": _case_copied,
    "_EinsumValues": _case_einsum,
    "_SparseLinearFM": _case_linear_fm,
    "_FastTranspose": _case_fast_transpose,
}


def _output_of(result):
    """What the Function returned, and the dense result it makes, read after a write."""
    if type(result) is STensor:
        return result.values, lambda: result.to_torch(in_place=False)
    return result, lambda: result


def _reference_of(result, dense):
    """The reference counterpart of ``_output_of(result)``: for an STensor, the dense
    reference's values at the result's coordinates, in its storage order."""
    if type(result) is STensor:
        keys = stensor_grad._Structure.of(result).keys()
        values = dense.reshape(-1)[keys]
        return values, lambda: _logical(result, values)
    return dense, lambda: dense


def _tolerance(dtype):
    if dtype == F64:
        return dict(rtol=1e-10, atol=1e-10)
    return dict(rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize("name", sorted(WRITE_CASES))
def test_a_functions_output_takes_in_place_writes(name, write):
    node, leaves, compute, reference = WRITE_CASES[name]()
    result = compute(*leaves)
    output, dense = _output_of(result)
    assert output.grad_fn.name() == node
    _write(write, output)
    got = dense()

    leaves64 = [leaf.detach().to(F64).requires_grad_() for leaf in leaves]
    ref_output, ref_dense = _reference_of(result, reference(*leaves64))
    _write(write, ref_output)
    want = ref_dense()
    g = torch.rand(want.shape, generator=torch.Generator().manual_seed(80), dtype=F64)
    (got * g.to(got.dtype)).sum().backward()
    (want * g).sum().backward()

    tol = _tolerance(got.dtype)
    torch.testing.assert_close(got.detach().to(F64), want.detach(), **tol)
    for leaf, leaf64 in zip(leaves, leaves64):
        torch.testing.assert_close(leaf.grad.to(F64), leaf64.grad, **tol)


@pytest.mark.parametrize("write", WRITES)
def test_fast_transpose_of_float64_is_torchs_copy_and_takes_in_place_writes(write):
    """The float64 twin of the _FastTranspose case: the native kernel is float32-only
    (scorch_transpose_2d_float), so fast_transpose returns ``x.T.contiguous()`` itself.
    The output is torch's copy, recorded by torch (CloneBackward0), and takes the same
    in-place writes with torch's own values and gradients, bit for bit."""
    (x,) = [_signed_dense(7, 5, seed=79)]
    output = scorch.fast_transpose(x)
    assert output.grad_fn.name() == "CloneBackward0"
    _write(write, output)

    x64 = x.detach().clone().requires_grad_()
    want = x64.t().contiguous()
    _write(write, want)
    g = torch.rand(want.shape, generator=torch.Generator().manual_seed(80), dtype=F64)
    (output * g).sum().backward()
    (want * g).sum().backward()

    torch.testing.assert_close(output.detach(), want.detach(), atol=0, rtol=0)
    torch.testing.assert_close(x.grad, x64.grad, atol=0, rtol=0)


@pytest.mark.parametrize("name", sorted(WRITE_CASES))
def test_a_functions_gradient_takes_in_place_writes_and_differentiates_again(name):
    """create_graph=True: the first gradient, written in place in grad mode (relu_),
    then differentiated with respect to every leaf, as dense float64 torch does."""
    node, leaves, compute, reference = WRITE_CASES[name]()
    got = _output_of(compute(*leaves))[1]()
    leaves64 = [leaf.detach().to(F64).requires_grad_() for leaf in leaves]
    want = reference(*leaves64)
    g = torch.rand(want.shape, generator=torch.Generator().manual_seed(81), dtype=F64)

    def second(dense, leaves, dtype):
        firsts = torch.autograd.grad(
            (dense.pow(2) * g.to(dtype)).sum(), leaves, create_graph=True
        )
        first = firsts[0]
        first.relu_()
        h = torch.rand(first.shape, generator=torch.Generator().manual_seed(82))
        seconds = torch.autograd.grad(
            (first * h.to(dtype)).sum(), leaves, allow_unused=True
        )
        return [first] + [
            torch.zeros_like(leaf) if grad is None else grad
            for grad, leaf in zip(seconds, leaves)
        ]

    tol = _tolerance(got.dtype)
    for actual, expected in zip(
        second(got, leaves, got.dtype), second(want, leaves64, F64)
    ):
        torch.testing.assert_close(actual.detach().to(F64), expected.detach(), **tol)


@pytest.mark.parametrize("write", WRITES)
@pytest.mark.parametrize("kind", ["torch result", "STensor result"])
def test_a_result_with_no_derivative_takes_in_place_writes(kind, write):
    """not_differentiable's output takes either write, and its backward still raises
    the coded refusal, with or without a graph asked of it."""
    x = _signed_dense(3, 4, seed=83)
    if kind == "torch result":
        result = x.detach() * 2
    else:
        result = _layout(x.detach() * 2, "ds")
    out = scorch_autograd.not_differentiable(
        result, "no derivative", "matmul_derivative_unsupported", (x,)
    )
    output, dense = _output_of(out)
    assert output.grad_fn.name() == "_NotDifferentiableBackward"
    _write(write, output)
    expected = x.detach() * 2
    _write(write, expected)
    torch.testing.assert_close(dense().detach(), expected)
    for create_graph in (False, True):
        with pytest.raises(MatmulDerivativeError) as raised:
            # retain_graph: the first backward frees relu_'s saved result before it raises
            torch.autograd.grad(
                dense().sum(), (x,), retain_graph=True, create_graph=create_graph
            )
        assert raised.value.code == "matmul_derivative_unsupported"


def test_compile_chain_with_parameters():
    dense = _dense((6, 5), seed=64, dtype=torch.float32)
    adj = scorch.from_csr(dense.to_sparse_csr(), "adj")
    adj.requires_grad = True
    x = torch.rand(5, 3, dtype=torch.float32, requires_grad=True)
    bias = torch.nn.Parameter(torch.rand(3, dtype=torch.float32))

    @scorch.compile
    def layer(adj, x, bias):
        return torch.relu(scorch.matmul(adj, x, format="dd") + bias)

    out = layer(adj, x, bias)
    out.sum().backward()
    w64 = dense.to(F64).requires_grad_()
    x64 = x.detach().to(F64).requires_grad_()
    b64 = bias.detach().to(F64).requires_grad_()
    torch.relu(w64 @ x64 + b64).sum().backward()
    keys = stensor_grad._Structure.of(adj).keys()
    torch.testing.assert_close(
        adj.grad.values.to(F64), w64.grad.reshape(-1)[keys], rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(x.grad.to(F64), x64.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(bias.grad.to(F64), b64.grad, rtol=1e-4, atol=1e-5)


# -- the training loop (scorch-merge, 12:3x) ----------------------------------------


def test_hand_written_training_loop_matches_dense_torch():
    dense = _dense((6, 5), density=0.5, seed=70)
    mask = (dense != 0).to(F64)
    a = scorch.from_torch(dense.clone())
    a.requires_grad = True
    a.to_sparse("ds")  # converted in place: the leaf stays the root
    reference = dense.clone().requires_grad_()
    x = torch.rand(5, 4, dtype=F64)
    target = torch.rand(6, 4, dtype=F64)
    lr = 0.1
    for _ in range(4):
        loss = (scorch.matmul(a, x) - target).pow(2).sum()
        loss.backward()
        ref_loss = ((reference * mask) @ x - target).pow(2).sum()
        ref_loss.backward()
        _close(loss.detach(), ref_loss.detach())
        with torch.no_grad():
            a.values -= lr * a.grad.values
            reference -= lr * reference.grad * mask
        a.grad = None
        reference.grad = None
        _check(a)
        assert a.is_leaf
        _close(a.to_torch(in_place=False).detach(), reference.detach() * mask)


# -- attributes and setters ---------------------------------------------------------


def test_values_and_grad_setters():
    template = _layout(_dense((4, 5), seed=80), "ds")
    a = _over(template, template._raw_values.clone())
    a.requires_grad = True
    same = a.values
    a.values = same  # what ``a.values -= x`` does after the in-place subtraction
    with pytest.raises(ScorchError) as raised:
        a.values = torch.zeros_like(same)
    assert raised.value.code == "stensor_values_assignment"
    scorch.matmul(a, torch.rand(5, 2, dtype=F64)).sum().backward()
    assert a.grad is not None
    a.grad = None
    assert a.grad is None
    b = a + a
    with pytest.raises(ScorchError) as raised:
        b.grad = None
    assert raised.value.code == "stensor_grad_not_leaf"
    with pytest.raises(ScorchError) as raised:
        b.requires_grad = False
    assert raised.value.code == "stensor_requires_grad_non_leaf"
    detached = b.detach()
    assert not detached.requires_grad
    assert detached._raw_values.data_ptr() == b._raw_values.data_ptr()


def test_pickle_and_deepcopy():
    template = _layout(_dense((4, 5), seed=81), "ds")
    a = _over(template, template._raw_values.clone())
    a.requires_grad = True
    for copied in (pickle.loads(pickle.dumps(a)), copy.deepcopy(a)):
        _check(copied)
        assert copied.requires_grad and copied.is_leaf
        assert torch.equal(copied._raw_values, a._raw_values)
        assert copied._raw_values.data_ptr() != a._raw_values.data_ptr()
    non_leaf = a + a
    restored = pickle.loads(pickle.dumps(non_leaf))
    assert restored.requires_grad and restored.is_leaf
    with pytest.raises(ScorchError) as raised:
        copy.deepcopy(non_leaf)
    assert raised.value.code == "stensor_deepcopy_non_leaf"


# -- the plan invariant and the internal carriers -----------------------------------


def test_no_plan_serves_a_grad_mode_call_on_a_tracked_stensor():
    dense = _dense((64, 48), density=0.1, seed=90)
    a = scorch.from_csr(dense.to_sparse_csr(), "a")
    b = torch.rand(48, 8, dtype=F64)
    with torch.no_grad():
        for _ in range(8):
            scorch.matmul(a, b)  # a plan, where the installer makes one
    a.requires_grad = True
    out = scorch.matmul(a, b)
    assert out.grad_fn is not None
    with torch.no_grad():
        assert scorch.matmul(a, b).grad_fn is None
    out.sum().backward()
    assert a.grad is not None
    x = torch.rand(3, 64, dtype=F64, requires_grad=True)
    scorch.matmul(x, a).sum().backward()
    structure = a.__dict__.get(GRAD_ATTR)
    if structure is not None:
        carriers = list(structure._carriers.values())
        if structure._transpose is not None:
            carriers += list(structure._transpose[0]._carriers.values())
        for carrier, _ in carriers:
            assert AUTOGRAD_VALUES not in carrier.__dict__


def _tensors_in(item, seen):
    if id(item) in seen:
        return
    seen.add(id(item))
    if isinstance(item, torch.Tensor):
        yield item
    elif isinstance(item, STensor):
        yield from _tensors_in(item.__dict__, seen)
    elif isinstance(item, dict):
        for key, value in item.items():
            yield from _tensors_in(key, seen)
            yield from _tensors_in(value, seen)
    elif isinstance(item, (list, tuple, set, frozenset)):
        for value in item:
            yield from _tensors_in(value, seen)


def _operand_copy_memo(monkeypatch, memo):
    """The operand-copy memo on or off, empty, and with its withdrawal counter at 0."""
    monkeypatch.setattr(stensor_module, "_MEMO_OPERAND_COPY", [memo])
    monkeypatch.setattr(stensor_module, "_OPERAND_COPY_CACHE", {})
    monkeypatch.setattr(
        stensor_module,
        "_OPERAND_COPY_STATE",
        [0, stensor_module._OPERAND_COPY_CACHE_MAX],
    )


def test_caches_hold_no_history(monkeypatch):
    # The operand-copy memo starts empty and on, so the two calls below and the
    # transposed float64 leaf after them all fill the same dict: the memo is filled here,
    # not only by whichever earlier test happened to fill it (b5 84322: the entry this
    # test caught came from an earlier test in the same process).
    _operand_copy_memo(monkeypatch, True)
    test_hand_written_training_loop_matches_dense_torch()
    test_einsum_through_the_compiler(EINSUM_PROGRAMS[1])
    leaf = torch.rand(6, 5, dtype=F64, requires_grad=True)
    scorch.from_torch(leaf.t())
    assert len(stensor_module._OPERAND_COPY_CACHE) >= 1
    caches = [
        (name, getattr(stensor_module, name))
        for name in ("_OPERAND_COPY_CACHE", "_RESULT_PARTS_CACHE", "_DENSE_PARTS_CACHE")
    ] + kernel_caches()
    for name, cache in caches:
        for tensor in _tensors_in(cache, set()):
            assert not tensor.requires_grad, name


def _transposed_step(w, g):
    """One step through ``from_torch(w.t())``: its value and ``w``'s gradient of
    ``sum(g * value**2)``, which depends on the values, so a stale copy or an old graph
    shows in both."""
    w.grad = None
    value = scorch.from_torch(w.t()).to_torch()
    (g * value**2).sum().backward()
    return value.detach().clone(), w.grad.clone()


def _transposed_reference(w):
    """The same step in dense float64 torch, from ``w``'s current values."""
    w64 = w.detach().to(F64).requires_grad_()
    g = torch.rand(5, 6, generator=torch.Generator().manual_seed(82), dtype=F64)
    (g * w64.t() ** 2).sum().backward()
    return w64.t().detach(), w64.grad, g


@pytest.mark.parametrize("dtype", [torch.float32, F64])
def test_a_remembered_operand_copy_gives_the_fresh_copys_gradient(monkeypatch, dtype):
    """Two steps on an unchanged leaf: the second ``from_torch(w.t())`` is served from the
    operand-copy memo.  Its value and gradient must equal the first step's and a memo-off
    step's exactly, and dense float64 torch's; the memo's copy carries no history."""
    w = torch.rand(6, 5, generator=torch.Generator().manual_seed(83), dtype=dtype)
    w.requires_grad_()
    want, want_grad, g = _transposed_reference(w)
    g = g.to(dtype)

    _operand_copy_memo(monkeypatch, True)
    first = _transposed_step(w, g)
    (entry,) = stensor_module._OPERAND_COPY_CACHE.values()
    second = _transposed_step(w, g)
    (again,) = stensor_module._OPERAND_COPY_CACHE.values()
    assert again[2] is entry[2]  # the second step was served from the memo
    assert not entry[2].requires_grad

    _operand_copy_memo(monkeypatch, False)
    fresh = _transposed_step(w, g)
    assert stensor_module._OPERAND_COPY_CACHE == {}

    for value, grad in (second, fresh):
        torch.testing.assert_close(value, first[0], atol=0, rtol=0)
        torch.testing.assert_close(grad, first[1], atol=0, rtol=0)
    tol = _tolerance(dtype)
    torch.testing.assert_close(first[0].to(F64), want, **tol)
    torch.testing.assert_close(first[1].to(F64), want_grad, **tol)


@pytest.mark.parametrize("dtype", [torch.float32, F64])
def test_a_no_grad_write_between_steps_gives_the_written_leafs_gradient(
    monkeypatch, dtype
):
    """A step, an optimizer-style write to the leaf (add_ under no_grad), a second step:
    the second value and gradient are the written leaf's, as dense float64 torch computes
    them from its current values -- not the remembered copy's, nor its old graph's."""
    w = torch.rand(6, 5, generator=torch.Generator().manual_seed(84), dtype=dtype)
    w.requires_grad_()
    _operand_copy_memo(monkeypatch, True)
    g = _transposed_reference(w)[2].to(dtype)
    _transposed_step(w, g)
    with torch.no_grad():
        w.add_(0.5)
    value, grad = _transposed_step(w, g)

    want, want_grad, _ = _transposed_reference(w)
    tol = _tolerance(dtype)
    torch.testing.assert_close(value.to(F64), want, **tol)
    torch.testing.assert_close(grad.to(F64), want_grad, **tol)


@pytest.mark.parametrize("dtype", [torch.float32, F64])
@pytest.mark.parametrize(
    "view",
    [lambda w: w, lambda w: w.t(), lambda w: torch._neg_view(w)],
    ids=["contiguous", "transposed", "neg_view"],
)
def test_from_torch_of_a_leaf_differentiates_on_every_values_path(
    monkeypatch, view, dtype
):
    """One case per return of stensor._flat_contiguous_values: the caller's own buffer
    (contiguous), a copy the operand-copy memo remembers (transposed), a resolved copy
    (a neg view).  Each stores detached values; the history is attached afterwards by
    _dense_from_torch's link_dense_copy, from the caller's tensor.  Through
    ``from_torch(...).to_torch()`` the value and the gradient are torch's, bit for bit.
    """
    _operand_copy_memo(monkeypatch, True)
    w = torch.rand(6, 5, generator=torch.Generator().manual_seed(85), dtype=dtype)
    w.requires_grad_()
    value = scorch.from_torch(view(w)).to_torch()
    g = torch.rand(
        value.shape, generator=torch.Generator().manual_seed(86), dtype=dtype
    )
    (value * g).sum().backward()

    w_ref = w.detach().clone().requires_grad_()
    want = view(w_ref)
    (want * g).sum().backward()
    torch.testing.assert_close(value.detach(), want.detach(), atol=0, rtol=0)
    torch.testing.assert_close(w.grad, w_ref.grad, atol=0, rtol=0)


def test_only_the_autograd_modules_name_the_tracked_values():
    root = pathlib.Path(scorch.__file__).parent
    allowed = {"stensor.py", "stensor_grad.py", "autograd.py", "einsum_grad.py"}
    pattern = re.compile(r"_autograd_values|AUTOGRAD_VALUES")
    scanned = set()
    for path in root.rglob("*.py"):
        name = path.relative_to(root).as_posix()
        scanned.add(name)
        if path.name not in allowed:
            assert not pattern.search(path.read_text()), path
    # The modules that hand tensors to native code, the compiler's LoopIR ones included.
    assert {
        "ops.py",
        "trace.py",
        "tiling.py",
        "prebuilt_kernels.py",
        "plan.py",
        "compiler/loopir/pipeline.py",
        "compiler/loopir/levels.py",
        "compiler/loopir/oracle.py",
        "compiler/loopir_spike/levels.py",
        "compiler/loopir_spike/interp.py",
    } <= scanned


def test_loopir_execution_on_tracked_operands_hands_native_code_their_storage():
    """LoopIR has no release callers and records no derivative; what it reads from a
    tracked STensor is the detached storage (the fixture above checks the native calls
    and the ``values`` readers)."""

    from scorch.compiler.loopir.pipeline import execute_cin_via_loopir

    from tests.test_scorch.test_loopir_pipeline_execution import (
        build_spmm_csr_dense,
        build_spmv_csr,
    )

    template = _layout(_dense((6, 5), seed=95), "ds")
    v = _start(template)
    a = _over(template, v)
    x = torch.rand(5, dtype=F64, requires_grad=True)
    b = torch.rand(5, 3, dtype=F64, requires_grad=True)
    for cin, dense, shape in (
        (build_spmv_csr(F64), x, (6,)),
        (build_spmm_csr_dense(dtype=F64), b, (6, 3)),
    ):
        operand = scorch.from_torch(dense)
        _check(operand)
        result, _ = execute_cin_via_loopir(cin, shape, a, operand)
        assert not result.requires_grad
        expected = (_logical(template, v) @ dense).detach()
        _close(result.to_torch(in_place=False), expected)
    _check(a)
