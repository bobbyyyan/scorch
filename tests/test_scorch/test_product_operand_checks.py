"""Operands that cannot be multiplied, refused by code on every route (patches 0074, 0074b).

A generated kernel bounds the contracted loop by one operand's extent and reads the
other operand by it.  ``matmul``'s matrix-vector branch and ``scorch.ops.spmv`` never
compared the two, so a vector shorter than the matrix's width was read past its end: the
call returned a wrong answer, or, read far enough past the end, ended the process with
SIGSEGV.  The prebuilt kernels refused a mismatch from their native checks with an
uncoded ``RuntimeError``, an STensor pair of dense layouts with torch's, and ``spmv``
refused a 2-D vector at the CIN verifier's stage boundary.  Now each mistake is refused
the same way on every route, before anything is built or any operand is read: by the
prebuilt kernel's own entry check, which carries the code; by einsum's or spmv's check
before a kernel is built; or, for two dense layouts, as torch's refusal under the same
class.  matmul itself checks nothing up front, so a correct call pays for no check:

1. ``matmul``: every operand kind, mistake and route (a fresh operand; an operand whose
   product is already planned or whose kernel is already kept; a gradient wanted; a
   keyword argument; ``use_cache=False``; the Python function the C entry hands calls
   to) raises ``IndexExtentMismatchError`` -- code ``einsum_index_extent_mismatch``, a
   ``CompileSpecError`` and a ``RuntimeError`` -- builds nothing, and leaves the
   operand's plans and kernels serving its correct products.
2. Two dense torch tensors are torch's product, and keep torch's own error.
3. An operand that is not a tensor is refused with ``matmul_operand_type_invalid``, a
   ``TypeError``, in either position.
4. ``spmv``: a vector of the wrong length, a 2-D vector, a 1-D matrix and an operand that
   is not an STensor, each with ``einsum``'s code for the same mistake.
5. ``einsum``'s own extent refusal raises the same class.
6. Every prebuilt kernel (SpMV, SpMM, SpGEMM; every value dtype and index width it takes)
   refuses each mismatch -- either operand short or long, the result's extents -- in its
   native check, and ``raise_if_native_coded`` gives that refusal the class and code.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import pytest
import torch

import scorch
from scorch import ops
from scorch import plan as plan_mod
from scorch.compiler.diagnostics import VerificationError
import scorch_ops
from scorch.exceptions import (
    CompileSpecError,
    IndexExtentMismatchError,
    TensorTypeError,
    defect_code,
    raise_if_native_coded,
)
from scorch.prebuilt_kernels import _MATMUL_PREBUILT_SPECS, resolve_prebuilt_matmul
from scorch.stensor import STensor

EXTENT = "einsum_index_extent_mismatch"
ROWS, COLS, WIDTH = 7, 5, 3
# Kinds of sparse operand.  The first three have plans; the STensor layouts after them
# reach the generated kernels (and "dd", torch's dense product).
PLANNED_KINDS = ["stensor_ds", "torch_csr64", "torch_csr32"]
KINDS = PLANNED_KINDS + ["stensor_ss", "stensor_sd", "stensor_oo", "stensor_dd", "torch_coo"]
# Each mistake: (its name, the correct product of that form, the product it gets wrong).
MISTAKES = ["A@v short", "A@v long", "A@X short", "A@X long", "X@A", "v@A"]
ROUTES = ["fresh", "warm", "grad", "keyword", "no_cache", "python"]


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


def _dense(dtype: torch.dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    dense = (torch.rand(ROWS, COLS, generator=g) < 0.4) * torch.rand(ROWS, COLS, generator=g)
    dense[0, COLS - 1] = 1.0  # the last column is read
    dense[ROWS - 1, 0] = 1.0  # and the last row
    return dense.to(dtype)


def _operand(kind: str, dense: torch.Tensor):
    if kind.startswith("stensor_"):
        return STensor.from_torch(dense).to_sparse(kind[len("stensor_"):])
    if kind == "torch_coo":
        return dense.to_sparse()
    csr = dense.to_sparse_csr()
    index = torch.int64 if kind == "torch_csr64" else torch.int32
    return torch.sparse_csr_tensor(
        csr.crow_indices().to(index), csr.col_indices().to(index), csr.values(),
        size=dense.shape,
    )


def _dense_operands(dtype: torch.dtype) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Per mistake: the dense operand of the correct product, and of the wrong one."""
    g = torch.Generator().manual_seed(9)

    def r(*shape):
        return torch.rand(*shape, generator=g, dtype=dtype)

    return {
        "A@v short": (r(COLS), r(COLS - 1)),
        "A@v long": (r(COLS), r(COLS + 1)),
        "A@X short": (r(COLS, WIDTH), r(COLS - 1, WIDTH)),
        "A@X long": (r(COLS, WIDTH), r(COLS + 1, WIDTH)),
        "X@A": (r(WIDTH, ROWS), r(WIDTH, ROWS + 1)),
        "v@A": (r(ROWS), r(ROWS - 1)),
    }


def _product(mistake: str, a, x, fn=None, **kwargs):
    fn = scorch.matmul if fn is None else fn
    if mistake in ("X@A", "v@A"):
        return fn(x, a, **kwargs)
    return fn(a, x, **kwargs)


def _reference(mistake: str, dense: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    d, xd = dense.double(), x.detach().double()
    return xd @ d if mistake in ("X@A", "v@A") else d @ xd


def _as_tensor(result):
    return result if isinstance(result, torch.Tensor) else result.to_torch(False)


def _chain(error: BaseException) -> List[BaseException]:
    seen, out, e = set(), [], error
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        out.append(e)
        e = e.__cause__ or e.__context__
    return out


def _assert_refused(call: Callable[[], object], cls, code: str, builds: List[int]):
    before = len(builds)
    with pytest.raises(BaseException) as caught:
        call()
    assert type(caught.value) is cls, repr(caught.value)
    assert defect_code(caught.value) == code
    assert not any(isinstance(e, VerificationError) for e in _chain(caught.value))
    assert len(builds) == before, "a refused call built a kernel"
    return caught.value


def _check_correct(mistake, a, dense, x, dtype):
    y = _as_tensor(_product(mistake, a, x))
    tol = 1e-6 if dtype == torch.float32 else 1e-13
    assert torch.allclose(y.detach().double(), _reference(mistake, dense, x), rtol=tol, atol=tol)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("mistake", MISTAKES)
@pytest.mark.parametrize("kind", KINDS)
def test_matmul_refuses_operands_that_cannot_be_multiplied(kind, mistake, route, builds):
    dtype = torch.float32
    dense = _dense(dtype)
    a = _operand(kind, dense)
    good, bad = _dense_operands(dtype)[mistake]
    if route == "warm":
        # The correct product, often enough that a plan serves it (or its kernel is
        # kept), then the mistake, then the correct product again.
        for _ in range(3):
            _check_correct(mistake, a, dense, good, dtype)
        if kind in PLANNED_KINDS and mistake in ("A@X short", "A@X long"):
            assert plan_mod.plans_of(a), "precondition: the product is planned"
        if kind.startswith("torch_csr") and mistake == "X@A":
            assert plan_mod.transposed_plans_of(a), "precondition: x @ A is planned"
    if route == "grad":
        bad = bad.clone().requires_grad_(True)
    call = {
        "fresh": lambda: _product(mistake, a, bad),
        "warm": lambda: _product(mistake, a, bad),
        "grad": lambda: _product(mistake, a, bad),
        "keyword": lambda: _product(mistake, a, bad, time_dict={}),
        "no_cache": lambda: _product(mistake, a, bad, use_cache=False),
        "python": lambda: _product(mistake, a, bad, fn=ops._matmul_python),
    }[route]
    error = _assert_refused(call, IndexExtentMismatchError, EXTENT, builds)
    assert isinstance(error, CompileSpecError) and isinstance(error, RuntimeError)
    if route == "warm":
        served = {key: plan.served for key, plan in plan_mod.plans_of(a).items()}
        _check_correct(mistake, a, dense, good, dtype)
        for key, plan in plan_mod.plans_of(a).items():
            if key in served:
                assert plan.served == served[key] + 1, "the plan stopped serving"


@pytest.mark.parametrize("mistake", ["A@v short", "A@X long", "X@A"])
@pytest.mark.parametrize("kind", PLANNED_KINDS)
def test_float64_planned_products_refuse_the_same_way(kind, mistake, builds):
    dense = _dense(torch.float64)
    a = _operand(kind, dense)
    good, bad = _dense_operands(torch.float64)[mistake]
    for _ in range(3):
        _check_correct(mistake, a, dense, good, torch.float64)
    _assert_refused(lambda: _product(mistake, a, bad), IndexExtentMismatchError, EXTENT, builds)
    _check_correct(mistake, a, dense, good, torch.float64)


def test_two_dense_tensors_keep_torchs_own_error():
    with pytest.raises(RuntimeError) as caught:
        scorch.matmul(torch.rand(ROWS, COLS), torch.rand(COLS + 1, WIDTH))
    assert not isinstance(caught.value, IndexExtentMismatchError)
    with pytest.raises(RuntimeError) as expected:
        torch.matmul(torch.rand(ROWS, COLS), torch.rand(COLS + 1, WIDTH))
    assert str(caught.value) == str(expected.value)


@pytest.mark.parametrize(
    "other",
    [[1.0] * COLS, 3.0, None, "x", object()],
    ids=["list", "float", "None", "str", "object"],
)
@pytest.mark.parametrize("kind", ["stensor_ds", "torch_csr64"])
def test_an_operand_that_is_not_a_tensor_is_refused(kind, other, builds):
    a = _operand(kind, _dense(torch.float32))
    for call in (lambda: scorch.matmul(a, other), lambda: scorch.matmul(other, a)):
        error = _assert_refused(call, TensorTypeError, "matmul_operand_type_invalid", builds)
        assert isinstance(error, TypeError)


def _spmv_cases():
    dense = _dense(torch.float32)
    st = STensor.from_torch(dense).to_sparse("ds")
    v = torch.rand(COLS)
    return [
        ("short vector", lambda: ops.spmv(st, STensor.from_torch(v[:-1])),
         IndexExtentMismatchError, EXTENT),
        ("long vector", lambda: ops.spmv(st, STensor.from_torch(torch.rand(COLS + 1))),
         IndexExtentMismatchError, EXTENT),
        ("2-D vector", lambda: ops.spmv(st, STensor.from_torch(torch.rand(COLS, 1))),
         CompileSpecError, "einsum_operand_rank_mismatch"),
        ("1-D matrix", lambda: ops.spmv(STensor.from_torch(v).to_sparse("s"), STensor.from_torch(v)),
         CompileSpecError, "einsum_operand_rank_mismatch"),
        ("torch vector", lambda: ops.spmv(st, v), TensorTypeError, "spmv_operand_type_invalid"),
        ("torch matrix", lambda: ops.spmv(dense.to_sparse_csr(), STensor.from_torch(v)),
         TensorTypeError, "spmv_operand_type_invalid"),
    ]


@pytest.mark.parametrize(
    "call, cls, code", [c[1:] for c in _spmv_cases()], ids=[c[0] for c in _spmv_cases()]
)
def test_spmv_refuses_operands_that_cannot_be_multiplied(call, cls, code, builds):
    _assert_refused(call, cls, code, builds)


def test_einsums_extent_refusal_is_the_same_class(builds):
    a = STensor.from_torch(_dense(torch.float32)).to_sparse("ds")
    error = _assert_refused(
        lambda: scorch.einsum("ij,j->i", a, torch.rand(COLS - 1)),
        IndexExtentMismatchError,
        EXTENT,
        builds,
    )
    assert str(error) == "einsum index 'j' has incompatible dimensions 5 and 4"


# The prebuilt kernels refuse a mismatch in their own native entry checks, with the code in
# the text (raise_if_native_coded), so each product they compute is refused there, by
# every value dtype and index width they take, before they read anything.
VALUE_DTYPES = {"f32": torch.float32, "f64": torch.float64, "i32": torch.int32, "i64": torch.int64}
INDEX_DTYPES = {"idx32": torch.int32, "idx64": torch.int64}
NATIVE_EXTENT = " [einsum_index_extent_mismatch]"


def _valued(rows: int, cols: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """Values a product of any of the four dtypes holds exactly."""
    g = torch.Generator().manual_seed(seed)
    dense = (torch.randint(1, 9, (rows, cols), generator=g) * (torch.rand(rows, cols, generator=g) < 0.5))
    dense[0, cols - 1] = 1  # the last column is read
    dense[rows - 1, 0] = 1  # and the last row
    return dense.to(dtype)


def _csr_stensor(dense: torch.Tensor, index: torch.dtype) -> STensor:
    csr = dense.to_sparse_csr()
    return STensor.from_torch(
        torch.sparse_csr_tensor(
            csr.crow_indices().to(index), csr.col_indices().to(index), csr.values(),
            size=dense.shape,
        )
    )


def _prebuilt_pair(product: str, dtype: torch.dtype, index: torch.dtype, a_cols: int, b_rows: int):
    """A CSR ``a`` of ``a_cols`` columns and, for ``product``, a second operand of ``b_rows`` rows."""
    a = _csr_stensor(_valued(ROWS, a_cols, dtype, 1), index)
    if product == "spmv":
        return a, _valued(b_rows, 1, dtype, 2).reshape(-1)
    if product == "spgemm":
        return a, _csr_stensor(_valued(b_rows, WIDTH, dtype, 2), index)
    return a, _valued(b_rows, WIDTH, dtype, 2)


# Each mismatch, as (A's columns, B's rows): B shorter or longer than A is wide, and A
# narrower or wider than B is tall.
EXTENT_MISTAKES = {
    "B short": (COLS, COLS - 1),
    "B long": (COLS, COLS + 1),
    "A narrow": (COLS - 1, COLS),
    "A wide": (COLS + 1, COLS),
}


@pytest.mark.parametrize("mistake", EXTENT_MISTAKES)
@pytest.mark.parametrize("index", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", VALUE_DTYPES)
@pytest.mark.parametrize("product", ["spmv", "spmm", "spgemm"])
def test_each_prebuilt_product_refuses_a_mismatch_in_its_kernel(
    product, dtype, index, mistake, builds
):
    a_cols, b_rows = EXTENT_MISTAKES[mistake]
    a, b = _prebuilt_pair(product, VALUE_DTYPES[dtype], INDEX_DTYPES[index], a_cols, b_rows)
    resolved = resolve_prebuilt_matmul(a, b if isinstance(b, STensor) else STensor.from_torch(b))
    assert resolved is not None, "precondition: a prebuilt kernel computes this product"
    error = _assert_refused(lambda: scorch.matmul(a, b), IndexExtentMismatchError, EXTENT, builds)
    # Refused by that kernel's own check: its name, the native line without the code, and
    # the native error kept as the cause.
    assert str(error) == (
        f"{resolved.symbol_name}: contraction mismatch: A.shape[1]={a_cols} "
        f"but B.shape[0]={b_rows}"
    )
    assert type(error.__cause__) is RuntimeError
    assert str(error.__cause__).splitlines()[0] == str(error) + NATIVE_EXTENT
    # The same operands, made to agree, still multiply correctly.
    good = _prebuilt_pair(product, VALUE_DTYPES[dtype], INDEX_DTYPES[index], COLS, COLS)
    y = _as_tensor(scorch.matmul(*good))
    expected = good[0].to_torch(False).double() @ _as_tensor(good[1]).double()
    assert torch.equal(y.double(), expected)


def _prebuilt_symbols() -> List[Tuple[str, torch.dtype, str, str]]:
    """Every prebuilt matmul symbol this build has: (symbol, dtype, lhs format, rhs format)."""
    out = []
    for spec in _MATMUL_PREBUILT_SPECS:
        for dtype, symbols in spec.symbol_by_dtype.items():
            for symbol in symbols:
                if getattr(scorch_ops, symbol, None) is not None:
                    out.append((symbol, dtype, spec.lhs_format, spec.rhs_format))
    return out


def _native_operand(fmt: str, dense: torch.Tensor, index: torch.dtype) -> STensor:
    if fmt == "d,s":
        return _csr_stensor(dense, index)
    if fmt == "o,o":
        return STensor.from_torch(dense).to_sparse("oo")
    return STensor.from_torch(dense)


def _result_mistakes(rank: int) -> Dict[str, Tuple[int, int, List[int]]]:
    """Each mismatch a kernel's arguments can carry, as (A's columns, B's rows, result shape)."""
    shape = [ROWS] if rank == 1 else [ROWS, WIDTH]
    out = {name: (a_cols, b_rows, shape) for name, (a_cols, b_rows) in EXTENT_MISTAKES.items()}
    out["result rows"] = (COLS, COLS, [ROWS + 1] + shape[1:])
    if rank == 2:
        out["result columns"] = (COLS, COLS, [ROWS, WIDTH - 1])
    return out


@pytest.mark.parametrize("index", INDEX_DTYPES)
@pytest.mark.parametrize(
    "symbol, dtype, lhs, rhs", _prebuilt_symbols(), ids=[s[0] for s in _prebuilt_symbols()]
)
def test_every_prebuilt_kernel_codes_every_extent_mismatch(symbol, dtype, lhs, rhs, index):
    """Called directly, so the arguments can carry a result shape matmul never builds."""
    if lhs == "o,o" and index == "idx64":
        pytest.skip("the COO operands here are built one way; the idx32 case runs them")
    fn = getattr(scorch_ops, symbol)
    rank = 1 if rhs == "d" else 2
    for name, (a_cols, b_rows, result_shape) in _result_mistakes(rank).items():
        a = _native_operand(lhs, _valued(ROWS, a_cols, dtype, 1), INDEX_DTYPES[index])
        b_dense = _valued(b_rows, 1 if rank == 1 else WIDTH, dtype, 2)
        b = (
            STensor.from_torch(b_dense.reshape(-1))
            if rank == 1
            else _native_operand(rhs, b_dense, INDEX_DTYPES[index])
        )
        args = [result_shape]
        for operand in (a, b):
            args += [list(operand.shape), operand._native_mode_indices(), operand._raw_values]
        with pytest.raises(IndexExtentMismatchError) as raised:
            try:
                fn(*args)
            except RuntimeError as error:
                raise_if_native_coded(error)
                raise
        assert defect_code(raised.value) == EXTENT, name
        assert str(raised.value).startswith(f"{symbol}: "), (name, str(raised.value))
        assert NATIVE_EXTENT not in str(raised.value), name
