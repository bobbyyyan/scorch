"""One keyword rule for ``scorch.matmul``, ``scorch.einsum`` and ``ops.spmv`` (patch 0072).

The three used to ignore any keyword they did not read, and several routes ignored ones
they did read elsewhere: ``einsum`` read only ``format=``, so ``output_format="ds"`` with a
sparse operand returned the inferred format; two dense torch tensors dropped every
keyword (``format="ds"`` returned a dense tensor, ``time_dict`` stayed empty); a dense
STensor pair read only ``format=``; ``format=`` silently won over a different
``output_format=``; ``out=`` was ignored on every route, leaving ``out`` unchanged; and
``output_mode_order`` was dropped by every route but the generated kernel's.  A dense
``format=`` on a product whose plain result is dense anyway made the call a keyword call,
which has no derivative and is never served by a plan.  Now, on every route:

1. A keyword the entry does not take raises ``KeywordArgumentError`` (a ``TypeError``, as
   Python raises), code ``unknown_keyword_argument``, before anything is built.
2. ``output_format=`` is ``format=``; both given and naming different formats is
   ``format_keywords_conflict``; naming the same format is accepted.
3. A keyword at its default value changes nothing: the call is the plain call.
4. A dense ``format=`` on a product whose plain result is a dense tensor, and an identity
   ``output_mode_order``, are the plain call: the same bits, a derivative when one is
   wanted, and a plan's service.
5. Any other ``output_mode_order`` is the result's storage order on every route, and
   changes nothing else: the result type is the plain call's.  Where the generated
   kernel refuses the order by code, the product in the default order is relaid out.
6. ``out=`` is torch's: filled and returned, with torch's refusals (dtype, gradient); a
   sparse result has no ``out=``.
"""

from __future__ import annotations

import re
import warnings
from typing import List

import pytest
import torch

import scorch
from scorch import ops
from scorch import plan as plan_mod
from scorch.exceptions import KeywordArgumentError, ScorchError, defect_code
from scorch.plan import plans_of
from scorch.stensor import STensor
from scorch.utils import parse_format

UNKNOWN = "unknown_keyword_argument"
CONFLICT = "format_keywords_conflict"
ROWS, COLS, WIDTH = 9, 7, 4


@pytest.fixture(autouse=True)
def plans_on():
    was_enabled = plan_mod.enabled()
    plan_mod.set_enabled(True)
    yield
    plan_mod.set_enabled(was_enabled)


@pytest.fixture
def builds(monkeypatch) -> List[int]:
    """Counts the builds of generated kernels; a build still runs."""
    count: List[int] = []
    build = ops._prepare_generated_kernel_build

    def counted(*args, **kwargs):
        count.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(ops, "_prepare_generated_kernel_build", counted)
    return count


def _dense(rows: int, cols: int, dtype=torch.float32, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    d = (torch.rand(rows, cols, generator=g) < 0.4) * (
        torch.rand(rows, cols, generator=g) + 0.5
    )
    d[0, cols - 1] = 1.0
    d[rows - 1, 0] = 1.0
    return d.to(dtype)


def _sparse(kind: str, dense: torch.Tensor):
    if kind.startswith("stensor_"):
        return STensor.from_torch(dense).to_sparse(kind[len("stensor_") :])
    if kind == "torch_csr":
        return dense.to_sparse_csr()
    if kind == "torch_coo":
        return dense.to_sparse()
    raise AssertionError(kind)


def _as_tensor(result):
    return result if isinstance(result, torch.Tensor) else result.to_torch(False)


def _logical(x):
    if isinstance(x, STensor):
        return x.to_torch(False)
    if isinstance(x, torch.Tensor) and x.layout is not torch.strided:
        return x.to_dense()
    return x


def _same(x, y):
    """The same type, format, mode order and bits."""
    assert type(x) is type(y), (type(x), type(y))
    if isinstance(x, STensor):
        assert x.format == y.format, (x.format, y.format)
        assert x.mode_order == y.mode_order, (x.mode_order, y.mode_order)
    else:
        assert x.stride() == y.stride(), (x.stride(), y.stride())
    torch.testing.assert_close(_as_tensor(x), _as_tensor(y), rtol=0, atol=0)


def _reference(a, b) -> torch.Tensor:
    """The product in float64, from the operands' dense values."""
    return _logical(a).double() @ _logical(b).double()


# Every route of matmul, as (name, a, b): the product's operands.
def _routes(dtype=torch.float32):
    A = _dense(ROWS, COLS, dtype)
    X = _dense(COLS, WIDTH, dtype, seed=5)
    v = _dense(COLS, 1, dtype, seed=7).reshape(-1)
    L = _dense(WIDTH, ROWS, dtype, seed=11)
    B = _dense(COLS, WIDTH, dtype, seed=13)
    return [
        ("two dense tensors", A, X),
        ("dense STensor pair", STensor.from_torch(A), STensor.from_torch(X)),
        ("prebuilt SpMM", _sparse("stensor_ds", A), X),
        ("generated SpMM", _sparse("stensor_ss", A), X),
        ("prebuilt SpMV", _sparse("stensor_ds", A), v),
        ("generated SpMV", _sparse("stensor_ss", A), v),
        ("torch CSR SpMM", _sparse("torch_csr", A), X),
        ("x @ A", L, _sparse("stensor_ds", A)),
        ("prebuilt SpGEMM", _sparse("stensor_ds", A), _sparse("stensor_ds", B)),
        ("COO x COO", _sparse("torch_coo", A), _sparse("torch_coo", B)),
    ]


ROUTE_NAMES = [r[0] for r in _routes()]
MATRIX_RESULT_ROUTES = [
    n for n in ROUTE_NAMES if "SpMV" not in n and n not in ("x @ A",)
] + ["x @ A"]


def _route(name, dtype=torch.float32):
    return next((a, b) for n, a, b in _routes(dtype) if n == name)


# 1. Unknown keywords.


@pytest.mark.parametrize("route", ROUTE_NAMES)
@pytest.mark.parametrize("keyword", ["fromat", "output", "dim", "compile_only"])
def test_matmul_refuses_an_unknown_keyword_on_every_route(route, keyword, builds):
    a, b = _route(route)
    with pytest.raises(KeywordArgumentError) as caught:
        scorch.matmul(a, b, **{keyword: 1})
    assert isinstance(caught.value, TypeError)
    assert defect_code(caught.value) == UNKNOWN
    assert f"matmul() got an unexpected keyword argument '{keyword}'" == str(
        caught.value
    )
    assert builds == [], "a refused call built a kernel"


@pytest.mark.parametrize("operands", ["dense", "sparse"])
@pytest.mark.parametrize("keyword", ["fromat", "out", "dim", "bogus"])
def test_einsum_refuses_an_unknown_keyword_dense_and_sparse(operands, keyword, builds):
    """out= among them: torch.einsum takes none, and raises TypeError for it."""
    A = _dense(ROWS, COLS)
    X = _dense(COLS, WIDTH, seed=5)
    a = A if operands == "dense" else _sparse("stensor_ds", A)
    with pytest.raises(KeywordArgumentError) as caught:
        scorch.einsum("ij,jk->ik", a, X, **{keyword: torch.empty(0)})
    assert isinstance(caught.value, TypeError)
    assert defect_code(caught.value) == UNKNOWN
    assert builds == []
    with pytest.raises(TypeError):
        torch.einsum("ij,jk->ik", A, X, **{keyword: torch.empty(0)})


@pytest.mark.parametrize("keyword", ["fromat", "out", "output_mode_order"])
def test_spmv_refuses_an_unknown_keyword(keyword, builds):
    a = _sparse("stensor_ss", _dense(ROWS, COLS))
    v = STensor.from_torch(_dense(COLS, 1, seed=7).reshape(-1))
    with pytest.raises(KeywordArgumentError) as caught:
        ops.spmv(a, v, **{keyword: [0]})
    assert defect_code(caught.value) == UNKNOWN
    assert builds == []


def test_the_accepted_sets():
    """The names each entry takes are the ones its routes read (0072's keyword scan)."""
    assert "out" in ops._MATMUL_KEYWORDS and "out" not in ops._EINSUM_KEYWORDS
    assert "compile_only" not in ops._MATMUL_KEYWORDS
    # Everything matmul hands to einsum (its einsum_kwargs, out= taken off), einsum takes.
    assert ops._MATMUL_KEYWORDS - {"out"} <= ops._EINSUM_KEYWORDS
    # Everything matmul hands to spmv, spmv takes (format= renamed to output_format).
    assert ops._MATMUL_KEYWORDS - {"out", "output_format", "output_mode_order"} <= (
        ops._SPMV_KEYWORDS
    )


# 2. output_format= is format=.

FORMATS = ["dd", "ds", ["d", "s"]]


def _for_rank(fmt, rank):
    if rank == 1:
        return "d" if fmt == "dd" else "s"
    return fmt


@pytest.mark.parametrize("route", ROUTE_NAMES)
@pytest.mark.parametrize("fmt", FORMATS, ids=str)
def test_output_format_is_format_on_every_matmul_route(route, fmt):
    a, b = _route(route)
    fmt = _for_rank(fmt, ops._matmul_result_rank(a, b))
    try:
        by_format = scorch.matmul(a, b, format=fmt)
    except ScorchError as error:
        # A format a route refuses is refused under every spelling, with one code.
        for spelling in ({"output_format": fmt}, {"format": fmt, "output_format": fmt}):
            with pytest.raises(type(error)) as caught:
                scorch.matmul(a, b, **spelling)
            assert defect_code(caught.value) == defect_code(error)
        return
    _same(scorch.matmul(a, b, output_format=fmt), by_format)
    _same(scorch.matmul(a, b, format=fmt, output_format=fmt), by_format)
    torch.testing.assert_close(
        _logical(by_format).double(), _reference(a, b), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("operands", ["dense", "sparse"])
@pytest.mark.parametrize("fmt", FORMATS, ids=str)
def test_output_format_is_format_on_einsum(operands, fmt):
    A = _dense(ROWS, COLS)
    X = _dense(COLS, WIDTH, seed=5)
    a = A if operands == "dense" else _sparse("stensor_ds", A)
    by_format = scorch.einsum("ij,jk->ik", a, X, format=fmt)
    _same(scorch.einsum("ij,jk->ik", a, X, output_format=fmt), by_format)


def test_sparse_einsum_output_format_returns_that_format():
    """L15 returned the inferred format ("dd" for CSR x dense) and ignored the request,
    a wrong result with no error (senior-engineer-a's case)."""
    a = _sparse("stensor_ds", _dense(ROWS, COLS))
    X = _dense(COLS, WIDTH, seed=5)
    result = scorch.einsum("ij,jk->ik", a, X, output_format="ds")
    assert isinstance(result, STensor)
    assert result.format == parse_format("ds")
    torch.testing.assert_close(
        _as_tensor(result).double(), _reference(a, X), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("fmt", ["d", "s"])
def test_spmv_format_is_output_format(fmt):
    a = _sparse("stensor_ss", _dense(ROWS, COLS))
    v = STensor.from_torch(_dense(COLS, 1, seed=7).reshape(-1))
    by_parameter = ops.spmv(a, v, output_format=fmt)
    _same(ops.spmv(a, v, format=fmt), by_parameter)
    _same(ops.spmv(a, v, fmt, format=[fmt]), by_parameter)
    with pytest.raises(KeywordArgumentError) as caught:
        ops.spmv(a, v, output_format="d", format="s")
    assert defect_code(caught.value) == CONFLICT


@pytest.mark.parametrize("entry", ["matmul", "einsum"])
def test_two_different_formats_are_refused(entry, builds):
    a = _sparse("stensor_ds", _dense(ROWS, COLS))
    X = _dense(COLS, WIDTH, seed=5)

    def call(**kw):
        if entry == "matmul":
            return scorch.matmul(a, X, **kw)
        return scorch.einsum("ij,jk->ik", a, X, **kw)

    with pytest.raises(KeywordArgumentError) as caught:
        call(format="dd", output_format="ds")
    assert isinstance(caught.value, TypeError)
    assert defect_code(caught.value) == CONFLICT
    assert builds == []
    # The same format spelt two ways is one format.
    _same(call(format="ds", output_format=["d", "s"]), call(format="ds"))


# Two dense operands honour every keyword.


@pytest.mark.parametrize("pair", ["two dense tensors", "dense STensor pair"])
def test_two_dense_operands_take_a_sparse_format_and_report_time(pair):
    a, b = _route(pair)
    result = scorch.matmul(a, b, format="ds")
    assert isinstance(result, STensor) and result.format == parse_format("ds")
    torch.testing.assert_close(
        _as_tensor(result).double(), _reference(a, b), atol=1e-5, rtol=1e-5
    )
    times = {}
    scorch.matmul(a, b, time_dict=times)
    assert "eval_time" in times


# 3 and 4. The plain call.

PLAIN_SPELLINGS = [
    {"format": "dd"},
    {"output_format": "dd"},
    {"format": ["d", "d"]},
    {"format": "dd", "output_format": "dd"},
    {"output_mode_order": [0, 1]},
    {"output_mode_order": (0, 1), "format": "dd"},
    {"format": None},
    {"use_cache": True},
    {"schedule": None},
    {"time_dict": None},
    {"out": None},
    {"format": "dd", "use_cache": True},
]


def _for_vector(kwargs):
    """The same spellings for a matrix times a vector."""
    out = {}
    for key, value in kwargs.items():
        if key == "output_mode_order":
            value = [0]
        elif value == "dd":
            value = "d"
        elif value == ["d", "d"]:
            value = ["d"]
        out[key] = value
    return out


@pytest.mark.parametrize(
    "kind", ["stensor_ds", "stensor_ss", "torch_csr", "stensor_oo"]
)
@pytest.mark.parametrize("kwargs", PLAIN_SPELLINGS, ids=str)
def test_a_keyword_that_changes_nothing_is_the_plain_call(kind, kwargs):
    dense = _dense(ROWS, COLS)
    a = _sparse(kind, dense)
    X = _dense(COLS, WIDTH, seed=5)
    _same(scorch.matmul(a, X, **kwargs), scorch.matmul(a, X))
    v = X[:, 0].contiguous()
    _same(scorch.matmul(a, v, **_for_vector(kwargs)), scorch.matmul(a, v))


@pytest.mark.parametrize("kwargs", PLAIN_SPELLINGS, ids=str)
def test_a_keyword_that_changes_nothing_keeps_the_derivative(kwargs):
    a = _sparse("stensor_ds", _dense(ROWS, COLS))
    X = _dense(COLS, WIDTH, seed=5)
    X1, X2 = X.clone().requires_grad_(True), X.clone().requires_grad_(True)
    y1 = scorch.matmul(a, X1, **kwargs)
    y2 = scorch.matmul(a, X2)
    assert y1.grad_fn is not None
    y1.sum().backward()
    y2.sum().backward()
    torch.testing.assert_close(X1.grad, X2.grad, rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", PLAIN_SPELLINGS, ids=str)
def test_a_keyword_that_changes_nothing_is_served_by_the_plan(kwargs):
    a = _sparse("stensor_ds", _dense(64, 64))
    X = _dense(64, 8, seed=5)
    for _ in range(3):
        scorch.matmul(a, X)
    (plan,) = plans_of(a).values()
    before = plan.served
    _same(scorch.matmul(a, X, **kwargs), scorch.matmul(a, X))
    assert plan.served == before + 2


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_a_dense_format_on_x_at_A_is_the_plain_call_where_autograd_computes_it(dtype):
    A = _sparse("stensor_ds", _dense(ROWS, COLS, dtype))
    L = _dense(WIDTH, ROWS, dtype, seed=11)
    _same(scorch.matmul(L, A, format="dd"), scorch.matmul(L, A))
    L1 = L.clone().requires_grad_(True)
    assert scorch.matmul(L1, A, format="dd").grad_fn is not None


def test_a_dense_format_on_integer_x_at_A_keeps_its_meaning():
    """For integer values autograd.py does not compute x @ A, and the plain call's result
    is a sparse STensor (einsum infers "ds" from A's compressed level); format="dd" asks
    for a dense result and gets one."""
    A = _sparse("stensor_ds", _dense(ROWS, COLS, torch.int64))
    L = _dense(WIDTH, ROWS, torch.int64, seed=11)
    assert not ops._autograd.computes(L, A)
    result = scorch.matmul(L, A, format="dd")
    assert isinstance(result, torch.Tensor) and result.layout == torch.strided
    torch.testing.assert_close(result, L @ A.to_torch(False))


def test_a_keyword_that_changes_the_result_still_bypasses_the_plan():
    a = _sparse("stensor_ds", _dense(64, 64))
    X = _dense(64, 8, seed=5)
    for _ in range(3):
        scorch.matmul(a, X)
    (plan,) = plans_of(a).values()
    before = plan.served
    times = {}
    scorch.matmul(a, X, time_dict=times)
    scorch.matmul(a, X, format="ds")
    scorch.matmul(a, X, output_mode_order=[1, 0])
    assert plan.served == before
    assert "eval_time" in times


# 5. output_mode_order.


def _stored_in(result, order):
    """Whether ``result`` stores its elements in mode order ``order``."""
    if isinstance(result, STensor):
        return list(result.mode_order) == list(order)
    return result.permute(*order).is_contiguous()


def _as_stensor(x):
    return x if isinstance(x, STensor) else STensor.from_torch(x)


IN_ORDER = "in order"


def _ordered(a, b, record_property, **kwargs):
    """``matmul(a, b, output_mode_order=[1, 0], **kwargs)`` and the route it took, which
    the test's JUnit record carries: ``IN_ORDER`` when the result was stored in that
    order as computed, else ``relayout:<code>``, when the generated kernel refused the
    order with that code and the product in the default order was relaid out
    (``ops._in_default_order``).  Before patch 0072 every route but the generated
    kernel's dropped the order and returned the product, so no such call may raise."""
    before = dict(ops._ORDER_RELAYOUTS)
    result = scorch.matmul(a, b, output_mode_order=[1, 0], **kwargs)
    codes = sorted(
        code
        for code, count in ops._ORDER_RELAYOUTS.items()
        if count != before.get(code, 0)
    )
    route = "relayout:" + ",".join(codes) if codes else IN_ORDER
    record_property("order_route", route)
    return result, route


@pytest.mark.parametrize("route", MATRIX_RESULT_ROUTES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_mode_order_is_the_results_storage_order(route, dtype, record_property):
    a, b = _route(route, dtype)
    record_property("operands", route)
    result, _ = _ordered(a, b, record_property)
    plain = scorch.matmul(a, b)
    assert type(result) is type(plain)
    assert _stored_in(result, [1, 0])
    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(
        _logical(result).double(), _reference(a, b), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize(
    "route", ["prebuilt SpGEMM", "two dense tensors", "prebuilt SpMM"]
)
def test_output_mode_order_with_a_sparse_format(route, record_property):
    """The order and format einsum gives the same request, on every route."""
    a, b = _route(route)
    record_property("operands", route)
    result, taken = _ordered(a, b, record_property, format="ds")
    assert isinstance(result, STensor)
    assert list(result.mode_order) == [1, 0]
    assert result.format == parse_format("ds")
    if taken == IN_ORDER:
        expected = scorch.einsum(
            "ij,jk->ik",
            _as_stensor(a),
            _as_stensor(b),
            format="ds",
            output_mode_order=[1, 0],
        )
        _same(result, expected)
    torch.testing.assert_close(
        _logical(result).double(), _reference(a, b), atol=1e-5, rtol=1e-5
    )


# Every operand kind matmul takes for a matrix: torch's dense, CSR and COO layouts, and
# STensors in each two-level format, with CSC as "ds" stored in mode order [1, 0].
OPERAND_KINDS = [
    "dense",
    "torch_csr",
    "torch_coo",
    "stensor_dd",
    "stensor_ds",
    "stensor_sd",
    "stensor_ss",
    "stensor_csc",
]
DENSE_KINDS = ("dense", "stensor_dd")


def _operand(kind: str, dense: torch.Tensor):
    if kind == "dense":
        return dense
    if kind == "stensor_csc":
        csc = STensor.from_torch(dense).to_sparse("ds")
        csc.change_mode_order([1, 0])
        return csc
    return _sparse(kind, dense)


# Every (kind_a, kind_b, result_format) but two dense operands with no format, which are
# not generated at all: test_dense_operand_* cover them.  The ids are the ones the three
# stacked parametrizations gave.
KIND_PAIR_CASES = [
    pytest.param(kind_a, kind_b, result_format, id=f"{kind_a}-{kind_b}-{result_format}")
    for result_format in [None, "ds", "sd", "ss"]
    for kind_b in OPERAND_KINDS
    for kind_a in OPERAND_KINDS
    if not (kind_a in DENSE_KINDS and kind_b in DENSE_KINDS and result_format is None)
]


@pytest.mark.parametrize("kind_a, kind_b, result_format", KIND_PAIR_CASES)
def test_output_mode_order_on_every_operand_kind_pair(
    kind_a, kind_b, result_format, record_property
):
    """output_mode_order [1, 0] on every pair of matrix operand kinds, with no format
    and with each sparse one: the result's storage order, its type and requested format
    the plain call's, its values float64's.  Where the plain call refuses the product by
    code, the call with the order is refused by code too, or computes it.  The JUnit
    record carries the route (``_ordered``), so the products whose order the compiler
    refuses can be counted by operand kinds."""
    a = _operand(kind_a, _dense(ROWS, COLS))
    b = _operand(kind_b, _dense(COLS, WIDTH, seed=5))
    kwargs = {} if result_format is None else {"format": result_format}
    record_property("operands", f"{kind_a} x {kind_b} format={result_format}")
    try:
        plain = scorch.matmul(a, b, **kwargs)
    except Exception as error:
        code = defect_code(error)
        assert code is not None, error
        record_property("plain_refused", code)
        plain = None
    try:
        result, _ = _ordered(a, b, record_property, **kwargs)
    except Exception as error:
        record_property("order_route", f"refused:{defect_code(error)}")
        assert plain is None, f"the plain call computes it; with the order: {error!r}"
        assert defect_code(error) is not None, error
        return
    if plain is not None:
        assert type(result) is type(plain)
    assert _stored_in(result, [1, 0])
    if result_format is not None:
        assert result.format == parse_format(result_format)
    torch.testing.assert_close(
        _logical(result).double(), _reference(a, b), atol=1e-5, rtol=1e-5
    )


def test_a_refused_order_is_computed_in_the_default_order_and_relaid_out(
    monkeypatch,
):
    """The generated kernel refusing the order by code: the product in the default order
    relaid out, counted under the refusal's code; an uncoded error still propagates."""
    from scorch.exceptions import ScorchNotImplementedError

    real = ops.einsum

    def refusing(expression, *operands, **kwargs):
        if "output_mode_order" in kwargs:
            raise ScorchNotImplementedError("order refused", code="test_order_refused")
        return real(expression, *operands, **kwargs)

    monkeypatch.setattr(ops, "einsum", refusing)
    a = _sparse("stensor_ds", _dense(ROWS, COLS, torch.float64))
    X = _dense(COLS, WIDTH, torch.float64, seed=5)
    before = ops._ORDER_RELAYOUTS.get("test_order_refused", 0)
    dense = scorch.matmul(a, X, output_mode_order=[1, 0])
    assert isinstance(dense, torch.Tensor) and _stored_in(dense, [1, 0])
    torch.testing.assert_close(dense, _reference(a, X), atol=1e-12, rtol=1e-12)
    sparse = scorch.matmul(a, X, output_mode_order=[1, 0], format="ss")
    assert isinstance(sparse, STensor) and list(sparse.mode_order) == [1, 0]
    assert sparse.format == parse_format("ss")
    torch.testing.assert_close(
        _logical(sparse), _reference(a, X), atol=1e-12, rtol=1e-12
    )
    assert ops._ORDER_RELAYOUTS["test_order_refused"] == before + 2

    def failing(expression, *operands, **kwargs):
        if "output_mode_order" in kwargs:
            raise RuntimeError("an uncoded failure")
        return real(expression, *operands, **kwargs)

    monkeypatch.setattr(ops, "einsum", failing)
    with pytest.raises(RuntimeError, match="an uncoded failure"):
        scorch.matmul(a, X, output_mode_order=[1, 0])
    assert ops._ORDER_RELAYOUTS["test_order_refused"] == before + 2


@pytest.mark.parametrize("use_cache", [True, False])
def test_output_mode_order_as_test_perf_large_passes_it(use_cache):
    """test_perf_large.py's three calls: format="dd", output_mode_order=[0, 1]."""
    a = _sparse("stensor_ds", _dense(64, 64))
    X = _dense(64, 8, seed=5)
    result = scorch.matmul(
        a, X, format="dd", use_cache=use_cache, output_mode_order=[0, 1]
    )
    _same(result, scorch.matmul(a, X, use_cache=use_cache))
    torch.testing.assert_close(result.double(), _reference(a, X), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "order,code",
    [
        ([1], "output_mode_order_not_permutation"),
        ([0, 0], "output_mode_order_not_permutation"),
        ("01", "output_mode_order_type_invalid"),
        ([0.0, 1.0], "output_mode_order_type_invalid"),
    ],
)
@pytest.mark.parametrize(
    "route", ["prebuilt SpMM", "two dense tensors", "prebuilt SpMV"]
)
def test_an_output_mode_order_that_is_no_permutation_is_refused(
    route, order, code, builds
):
    a, b = _route(route)
    with pytest.raises(ScorchError) as caught:
        scorch.matmul(a, b, output_mode_order=order)
    assert defect_code(caught.value) == code
    assert builds == []


# 6. out=.


@pytest.mark.parametrize(
    "route", [n for n in ROUTE_NAMES if "SpGEMM" not in n and "COO x COO" not in n]
)
def test_out_is_filled_and_returned(route):
    a, b = _route(route)
    expected = _as_tensor(scorch.matmul(a, b))
    out = torch.full_like(expected, -1.0)
    returned = scorch.matmul(a, b, out=out)
    assert returned is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "route", ["two dense tensors", "prebuilt SpMM", "generated SpMM", "x @ A"]
)
def test_out_is_resized_with_torch_warning(route):
    a, b = _route(route)
    expected = _as_tensor(scorch.matmul(a, b))
    out = torch.zeros(2, 3)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        scorch.matmul(a, b, out=out)
    assert out.shape == expected.shape
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert any("resized" in str(w.message) for w in seen), [
        str(w.message) for w in seen
    ]


@pytest.mark.parametrize(
    "route", ["two dense tensors", "prebuilt SpMM", "generated SpMV", "x @ A"]
)
def test_out_of_another_dtype_is_refused_as_torch_refuses(route):
    a, b = _route(route)
    out = torch.empty(0, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        torch.matmul(_dense(2, 2), _dense(2, 2), out=out)  # torch's own refusal
    with pytest.raises(RuntimeError):
        scorch.matmul(a, b, out=out)


@pytest.mark.parametrize("route", ["two dense tensors", "prebuilt SpMM", "x @ A"])
def test_out_with_a_gradient_wanted_is_refused_as_torch_refuses(route):
    a, b = _route(route)
    if isinstance(b, torch.Tensor):
        b = b.clone().requires_grad_(True)
    else:
        a = a.clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match="don't support automatic differentiation"):
        scorch.matmul(a, b, out=torch.empty(0))
    with torch.no_grad():
        scorch.matmul(a, b, out=torch.empty(0))  # allowed with no gradient wanted


# (name, make out, the result shape): the out= cases torch's rule distinguishes.
RESIZE_CASES = [
    ("same shape", lambda: torch.zeros(3, 4), (3, 4)),
    ("empty", lambda: torch.empty(0), (3, 4)),
    ("elements, another shape", lambda: torch.zeros(2, 3), (3, 4)),
    ("non-contiguous, same shape", lambda: torch.zeros(4, 3).t(), (3, 4)),
    ("non-contiguous, another shape", lambda: torch.zeros(6, 2).t(), (3, 4)),
]


@pytest.mark.parametrize("case", RESIZE_CASES, ids=[c[0] for c in RESIZE_CASES])
def test_resize_out_is_torchs_rule(case):
    """``ops._resize_out`` reimplements torch's rule for out= (``at::native::
    resize_output``; its operator, ``torch._resize_output_``, has no CPU kernel in torch
    2.13).  torch's own CPU out= path applies the rule: ``torch.matmul(a, b, out=theirs)``
    resizes ``theirs`` by it, warning included, before writing the product.  So the two
    must agree on the shape, strides and warnings, on any torch.  A wrong dtype or device
    is refused before either runs
    (``test_out_of_another_dtype_is_refused_as_torch_refuses``)."""
    _, make, shape = case
    a = torch.rand(shape[0], 2, dtype=torch.float32)
    b = torch.rand(2, shape[1], dtype=torch.float32)
    mine, theirs = make(), make()
    with warnings.catch_warnings(record=True) as mine_seen:
        warnings.simplefilter("always")
        ops._resize_out(mine, torch.Size(shape))
    with warnings.catch_warnings(record=True) as theirs_seen:
        warnings.simplefilter("always")
        torch.matmul(a, b, out=theirs)
    torch.testing.assert_close(theirs, a @ b)  # torch's out= path did write the product
    assert mine.shape == theirs.shape
    assert mine.stride() == theirs.stride()
    # A warning raised in C++ (TORCH_WARN) reaches Python with the source location
    # appended; the rule's text is what precedes it.
    located = re.compile(r"\s*\(Triggered internally at .*\)\s*$", re.DOTALL)
    assert [str(w.message) for w in mine_seen] == [
        located.sub("", str(w.message)) for w in theirs_seen
    ]
    assert [w.category for w in mine_seen] == [w.category for w in theirs_seen]


@pytest.mark.parametrize("route", ["prebuilt SpGEMM", "COO x COO", "prebuilt SpMM"])
def test_out_takes_no_sparse_result(route):
    a, b = _route(route)
    kwargs = {"format": "ds"} if route == "prebuilt SpMM" else {}
    with pytest.raises(ScorchError) as caught:
        scorch.matmul(a, b, out=torch.empty(0), **kwargs)
    assert defect_code(caught.value) == "out_requires_dense_result"


def test_out_takes_no_output_mode_order():
    a, b = _route("prebuilt SpMM")
    with pytest.raises(ScorchError) as caught:
        scorch.matmul(a, b, out=torch.empty(0), output_mode_order=[1, 0])
    assert defect_code(caught.value) == "out_requires_dense_result"


def test_out_may_overlap_an_operand_on_a_sparse_route():
    """The product is computed into a temporary first, so out= may be the dense operand
    itself."""
    a = _sparse("stensor_ds", _dense(COLS, COLS))
    X = _dense(COLS, WIDTH, seed=5)
    expected = scorch.matmul(a, X)
    scorch.matmul(a, X, out=X)
    torch.testing.assert_close(X, expected, rtol=0, atol=0)


# scorch.compile's fused path forwards matmul's keywords as the traced call made them.

COMPILE_SPELLINGS = [
    {},
    {"format": "dd"},
    {"output_format": "dd"},
    {"format": "dd", "output_format": "dd"},
    {"format": None},
]


@pytest.mark.parametrize("kwargs", COMPILE_SPELLINGS, ids=str)
def test_compile_fused_path_with_every_keyword_it_forwards(kwargs):
    """trace.py fuses a matmul whose keywords are format= and output_format= (its only
    accepted ones), through the native fused kernel or ops.matmul(a, b, **kwargs); both
    must agree with the eager product."""
    a = _sparse("stensor_ds", _dense(64, 32))
    X = _dense(32, 16, seed=5)
    bias = torch.rand(16, generator=torch.Generator().manual_seed(9))

    def layer(a, x, bias):
        return torch.relu(scorch.matmul(a, x, **kwargs) + bias)

    compiled = scorch.compile(layer)
    for _ in range(2):
        got = compiled(a, X, bias)
    want = torch.relu(_as_tensor(scorch.matmul(a, X)) + bias)
    torch.testing.assert_close(_as_tensor(got), want, atol=1e-5, rtol=1e-5)


def test_compile_refuses_two_different_formats():
    a = _sparse("stensor_ds", _dense(64, 32))
    X = _dense(32, 16, seed=5)
    bias = torch.rand(16)

    def layer(a, x, bias):
        return torch.relu(scorch.matmul(a, x, format="dd", output_format="ds") + bias)

    with pytest.raises(KeywordArgumentError) as caught:
        scorch.compile(layer)(a, X, bias)
    assert defect_code(caught.value) == CONFLICT
