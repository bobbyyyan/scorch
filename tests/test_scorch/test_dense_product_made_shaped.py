"""A dense SpMM product is made in its final shape and reaches the caller as it was made.

Every prebuilt kernel that computes a dense product allocates it as the ``[rows, k]`` matrix
it holds (csrc/spmm.h, the note on the v2 core's output; csrc/native_abi.h's row panels).
Nothing gives the product its shape afterwards, so it is a tensor of its own, as
torch.matmul's is, and a new operand's first call makes no shaping call. Before this, each
kernel returned a flat buffer and every caller shaped it: with ``reshape`` or ``view``, which
make a view (a custom autograd Function's forward must not return one, and
``scorch.compile``'s fused path and ``sparse_linear_fm`` returned one), or with ``resize_``
in place behind a use-count check, on every new operand.

Each test below fails on that code: the first two on the buffer's rank, the third on a view,
the fourth on the calls a profiler sees. The last pins autograd.py's calls on a
grad-mode step of a matrix product, per function, so that a helper layer added on that
path (that code had one, which shaped and copied each product after the kernel) fails it
by name.
"""

import collections
import gc
import os
import sys

import pytest
import torch

import scorch
import scorch_ops
from scorch import ops
from scorch.prebuilt_kernels import _MATMUL_PREBUILT_SPECS, resolve_prebuilt_fused
from scorch.stensor import STensor

# Odd extents, so that a flat buffer and the matrix differ in more than their rank.
ROWS, COLS, K = 37, 29, 5


def _dense_operand(dtype, seed=3, rows=ROWS, cols=COLS):
    """A sparse pattern with an empty row, as a dense tensor of ``dtype``."""
    g = torch.Generator().manual_seed(seed)
    keep = torch.rand(rows, cols, generator=g) < 0.25
    if dtype.is_floating_point:
        values = torch.rand(rows, cols, generator=g, dtype=torch.float64) + 0.5
    else:
        values = torch.randint(1, 5, (rows, cols), generator=g).double()
    dense = (values * keep).to(dtype)
    dense[5] = 0
    return dense


def _dense(rows, cols, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        return torch.rand(rows, cols, generator=g, dtype=torch.float64).to(dtype) - 0.5
    return torch.randint(-3, 4, (rows, cols), generator=g).to(dtype)


def _native_args(a_dense, b, lhs_format="d,s"):
    """The native kernels' arguments for ``a_dense @ b``, from the caller's own arrays."""
    rows, cols = a_dense.shape
    if lhs_format == "o,o":
        coo = a_dense.to_sparse_coo().coalesce()
        index = coo.indices().to(torch.int32)
        levels = [[index[0].contiguous()], [index[1].contiguous()]]
        values = coo.values()
    else:
        csr = a_dense.to_sparse_csr()
        levels = [
            [],
            [csr.crow_indices().to(torch.int32), csr.col_indices().to(torch.int32)],
        ]
        values = csr.values()
    return (
        [rows, b.shape[1]],
        [rows, cols],
        levels,
        values,
        list(b.shape),
        [[], []],
        b.reshape(-1),
    )


def _reference(a_dense, b):
    return (a_dense.double() @ b.double()).to(b.dtype)


def _check_matrix(out, expected):
    """The product as the matrix itself: its shape, contiguous, a tensor of its own, and right."""
    assert out.shape == expected.shape, f"rank or shape {tuple(out.shape)}"
    assert out.is_contiguous()
    assert out._base is None, "the product is a view"
    if expected.dtype.is_floating_point:
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    else:
        assert torch.equal(out, expected)


def _dense_output_symbols():
    """Every symbol ``resolve_prebuilt_matmul`` can serve a dense product with."""
    found = []
    for spec in _MATMUL_PREBUILT_SPECS:
        if spec.output_format != "dd":
            continue
        for dtype, symbols in spec.symbol_by_dtype.items():
            found.extend((spec.lhs_format, dtype, symbol) for symbol in symbols)
    return found


@pytest.mark.parametrize(
    "lhs_format, dtype, symbol",
    _dense_output_symbols(),
    ids=lambda v: str(v).replace("torch.", ""),
)
def test_every_dispatchable_dense_spmm_kernel_returns_the_matrix(
    lhs_format, dtype, symbol
):
    fn = getattr(scorch_ops, symbol)
    a_dense = _dense_operand(dtype)
    b = _dense(COLS, K, dtype, seed=4)
    out = fn(*_native_args(a_dense, b, lhs_format)).storage.value
    _check_matrix(out, _reference(a_dense, b))


# The older float32 SpMM kernels the extension still binds (benches and tests call them);
# matmul dispatches none of them, and each allocates its product as the matrix too.
OLDER_FLOAT_KERNELS = [
    "spmm_csr_float_" + name
    for name in (
        "untiled",
        "optimized",
        "turbo",
        "ultra",
        "apex",
        "tiled_i_k",
        "direct",
        "neon",
        "row_panel",
        "k_parallel",
        "sorted_rows",
        "neon2",
        "neon4",
        "tiled_neon",
    )
]


@pytest.mark.parametrize("symbol", OLDER_FLOAT_KERNELS)
def test_every_older_float_spmm_kernel_returns_the_matrix(symbol):
    fn = getattr(scorch_ops, symbol)
    a_dense = _dense_operand(torch.float32)
    b = _dense(COLS, K, torch.float32, seed=4)
    out = fn(*_native_args(a_dense, b)).storage.value
    _check_matrix(out, _reference(a_dense, b))


@pytest.fixture
def panel_limit():
    """Lowers the int32 row-panel bound for one test and restores it after (as
    test_csr_row_panels.py does)."""
    original = scorch_ops.abi_csr_panel_limit()
    yield scorch_ops.abi_csr_panel_limit
    scorch_ops.abi_csr_panel_limit(original)


@pytest.mark.parametrize(
    "kernel",
    ["tilej", "tileijk", "bias", "bias relu", "row panels f32", "row panels f64"],
)
def test_the_tiled_fused_and_panelled_kernels_return_the_matrix(kernel, panel_limit):
    dtype = torch.float64 if kernel == "row panels f64" else torch.float32
    a_dense = _dense_operand(dtype)
    b = _dense(COLS, K, dtype, seed=4)
    args = _native_args(a_dense, b)
    expected = _reference(a_dense, b)
    if kernel in ("tilej", "tileijk"):
        name = "spmm_csr_float_" + kernel
        if not hasattr(scorch_ops, name):
            pytest.skip(f"{name} is not built on this host")
        extra = (COLS // 4,) if kernel == "tilej" else (K // 2, COLS // 4)
        out = getattr(scorch_ops, name)(*args, *extra).storage.value
    elif kernel.startswith("bias"):
        kinds = ("add", "relu") if kernel == "bias relu" else ("add",)
        resolved = resolve_prebuilt_fused("d,s", "d,d", kinds, torch.float32)
        assert resolved is not None, f"no fused kernel for {kinds}"
        bias = _dense(1, K, dtype, seed=5).reshape(-1)
        out = resolved.fn(*args, bias).storage.value
        expected = expected + bias
        if kernel == "bias relu":
            expected = expected.relu()
    else:
        nnz = int(args[3].numel())
        panel_limit(nnz // 3)
        before = scorch_ops.abi_csr_panelled_calls()
        name = "spmm_csr_float_v2" if dtype == torch.float32 else "spmm_csr_double_v2"
        out = getattr(scorch_ops, name)(*args).storage.value
        assert scorch_ops.abi_csr_panelled_calls() == before + 1, "the call ran whole"
    _check_matrix(out, expected)


@scorch.compile
def _bias_relu(a, b, bias):
    return torch.relu(scorch.matmul(a, b) + bias)


# The two Python paths that took a flat buffer's ``view`` or ``reshape`` as their result.


def test_scorch_compile_s_fused_product_is_not_a_view():
    a_dense = _dense_operand(torch.float32)
    a = STensor.from_torch(a_dense.to_sparse_csr())
    b = _dense(COLS, K, torch.float32, seed=4)
    bias = _dense(1, K, torch.float32, seed=5).reshape(-1)
    fused = _bias_relu(a, b, bias)
    native = resolve_prebuilt_fused("d,s", "d,d", ("add", "relu"), torch.float32)
    # reshape: the same values whichever rank the kernel's buffer has
    expected = native.fn(*_native_args(a_dense, b), bias).storage.value.reshape(ROWS, K)
    assert torch.equal(fused, expected), "the fused kernel did not serve the call"
    assert fused._base is None, "scorch.compile's fused product is a view"


def test_sparse_linear_fm_s_product_is_not_a_view():
    a = STensor.from_torch(_dense_operand(torch.float32).to_sparse_csr())
    x_fm = _dense(COLS, K, torch.float32, seed=6)
    bias = _dense(1, ROWS, torch.float32, seed=7).reshape(-1)
    linear = ops.sparse_linear_fm(x_fm, a, bias=bias)
    assert linear.shape == (ROWS, K)
    torch.testing.assert_close(
        linear, _reference(a.to_torch(), x_fm) + bias[:, None], rtol=1e-5, atol=1e-5
    )
    assert linear._base is None, "sparse_linear_fm's product is a view"


# The Tensor methods that give a tensor another shape, in place or as a view.
_SHAPING = frozenset(
    {
        "reshape",
        "view",
        "view_as",
        "reshape_as",
        "resize_",
        "resize_as_",
        "as_strided",
        "as_strided_",
        "set_",
        "flatten",
        "unflatten",
        "squeeze",
        "unsqueeze",
        "detach",
    }
)


def _shaping_calls_on(call):
    """``call()``'s result, and the shaping calls made on its memory while it ran, as a
    profiler sees them: each Tensor method in ``_SHAPING`` called on a tensor over the
    result's storage."""
    seen = []

    def profile(frame, event, arg):
        if event == "c_call" and getattr(arg, "__name__", None) in _SHAPING:
            owner = getattr(arg, "__self__", None)
            if isinstance(owner, torch.Tensor):
                seen.append((arg.__name__, owner.untyped_storage().data_ptr()))

    sys.setprofile(profile)
    try:
        out = call()
    finally:
        sys.setprofile(None)
    where = out.untyped_storage().data_ptr()
    return out, [name for name, ptr in seen if ptr == where]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("operand", ["stensor", "torch csr"])
def test_a_prebuilt_product_takes_no_shaping_call(operand, dtype):
    """The first call on a new operand, the second, and one a plan serves: none gives the
    product a shape after the kernel made it (the second call shaped it in Python before).
    """
    a_dense = _dense_operand(dtype)
    csr = a_dense.to_sparse_csr()
    a = STensor.from_torch(csr) if operand == "stensor" else csr
    b = _dense(COLS, K, dtype, seed=4)
    expected = _reference(a_dense, b)
    for call in ("first", "second", "third"):
        out, shaping = _shaping_calls_on(lambda: scorch.matmul(a, b))
        assert shaping == [], f"{call} call: {shaping}"
        _check_matrix(out, expected)


# autograd.py's Python calls on one grad-mode step of a matrix product (the product, the
# backward of ``(y * upstream).sum()`` and the update), per function, on the steps a
# training loop makes: a new operand's first step, its second, and the step after 4 more
# (the steady one, which every later step repeats). The second step adds the dispatch's
# own calls, made before a plan serves the call; the first also builds the operand's
# structure and its transpose. The counts are the ones this code makes, so a call added
# on the path fails the test by name.
_STEADY = {
    "A @ x": {
        "_SparseMM.forward": 1,
        "_SparseMM.backward": 1,
        "_mm_backward": 1,
        "_in_layout": 2,
    },
    "x @ A": {
        "_SparseXA.forward": 1,
        "_SparseXA.backward": 1,
        "_mm_t_backward": 1,
        "_in_layout": 1,
        "_x_at_a": 1,
        "_transpose": 4,
        "is_sparse": 1,
    },
}
_EVERY_STEP = {
    "matmul": 1,
    "serve": 1,
    "wanted": 1,
    "is_sparse": 3,
    "_covered": 1,
    "_differentiable": 1,
    "_sparse_operand": 1,
    "_stensor_operand": 1,
    "_cached": 1,
    "_device_of": 1,
    "_dtype_of": 1,
    "_transposed_layout": 3,
    "_matmul": 1,
    "_plain": 2,
    "_record": 1,
    "_stamp": 2,
    "_Operand.__init__": 1,
    "_Operand._owns": 1,
    "_Operand.mm": 1,
    "_Operand.mm_t": 1,
    "_Structure.canonical": 1,
    "_Structure.describes": 1,
    "_Structure.describes.<locals>.<genexpr>": 3,
    "_Structure.mm": 1,
    "_Structure.transpose": 1,
}
_SECOND_ADDS = {"serve": 2, "wanted": 2, "is_sparse": 2, "_matmul": 1, "_nthreads": 1}
_FIRST_ADDS = {
    "_Structure.describes": -1,
    "_Structure.describes.<locals>.<genexpr>": -3,
    "_Structure.__init__": 2,
    "_Structure.__init__.<locals>.<genexpr>": 8,
    "_shared_transpose": 1,
    "_shared_transpose.<locals>.<genexpr>": 6,
    "_stensor_operand.<locals>.build": 1,
    "_stamp": 2,
}


def _pinned(route, step):
    calls = collections.Counter(_EVERY_STEP)
    calls.update(_STEADY[route])
    if step in ("second", "first"):
        calls.update(_SECOND_ADDS)
    if step == "first":
        calls.update(_FIRST_ADDS)
    return {name: n for name, n in calls.items() if n}


_AUTOGRAD_PY = os.path.join("scorch", "autograd.py")
# List, dict and set comprehensions are left out of the count: Python 3.12 inlines them
# into the function that holds them (PEP 709), so they are frames of their own, and
# calls, only on 3.11. Generator expressions are frames on both.
_INLINED = ("<listcomp>", "<dictcomp>", "<setcomp>")


def _autograd_calls(step):
    """autograd.py's Python calls while ``step()`` runs, by qualified name (a generator
    counts once per resumption), with the collector off so that it cannot add any."""
    calls = collections.Counter()

    def profile(frame, event, arg):
        code = frame.f_code
        if (
            event == "call"
            and code.co_filename.endswith(_AUTOGRAD_PY)
            and not code.co_qualname.endswith(_INLINED)
        ):
            calls[code.co_qualname] += 1

    gc.collect()
    gc.disable()
    sys.setprofile(profile)
    try:
        step()
    finally:
        sys.setprofile(None)
        gc.enable()
    return dict(calls)


# 64 rows of 8 stored entries each, and k = 8, in float32.
GRAD_N, GRAD_D, GRAD_K = 64, 8, 8


def _grad_step(route, seed):
    g = torch.Generator().manual_seed(seed)
    cols = torch.stack(
        [torch.randperm(GRAD_N, generator=g)[:GRAD_D] for _ in range(GRAD_N)]
    )
    crow = torch.arange(0, GRAD_N * GRAD_D + 1, GRAD_D, dtype=torch.int32)
    values = torch.rand(GRAD_N * GRAD_D, generator=g) + 0.5
    a = STensor.from_csr(
        torch.sparse_csr_tensor(
            crow,
            cols.sort(dim=1).values.reshape(-1).to(torch.int32),
            values,
            (GRAD_N,) * 2,
        )
    )
    shape = (GRAD_K, GRAD_N) if route == "x @ A" else (GRAD_N, GRAD_K)
    leaf = (torch.rand(shape, generator=g) - 0.5).requires_grad_()
    upstream = torch.rand(shape, generator=g) - 0.5

    def step():
        y = scorch.matmul(leaf, a) if route == "x @ A" else scorch.matmul(a, leaf)
        (y * upstream).sum().backward()
        with torch.no_grad():
            leaf.sub_(1e-4 * leaf.grad)
        leaf.grad = None

    return step


@pytest.mark.parametrize("route", ["A @ x", "x @ A"])
def test_a_matrix_grad_step_makes_only_the_pinned_autograd_calls(route):
    warm = _grad_step(route, seed=10)
    for _ in range(10):  # kernels, plans and the allocator warm
        warm()
    for seed in (11, 12):
        _grad_step(route, seed)()
    step = _grad_step(route, seed=13)
    made = {"first": _autograd_calls(step), "second": _autograd_calls(step)}
    for _ in range(4):
        step()
    made["steady"] = _autograd_calls(step)
    wrong = []
    for which, calls in made.items():
        pinned = _pinned(route, which)
        for name in sorted(set(pinned) | set(calls)):
            want, got = pinned.get(name, 0), calls.get(name, 0)
            if want != got:
                wrong.append(f"{which} step: {name} {want} -> {got}")
    assert not wrong, "autograd.py's calls, pinned -> made: " + "; ".join(wrong)
