"""matmul's dtype slow path: operands whose type is not exactly a tensor or an STensor.

``matmul`` admits three exact-type pairs inline: an STensor or a tensor times a tensor,
and an STensor times an STensor.  Every other pair takes the slow path,
``_require_one_product_dtype``, which refuses an operand that is not a tensor and a pair
of different value dtypes.  These tests hold the slow path to torch on the operands that
take it in a model:

1. ``torch.nn.Parameter`` operands, with and without ``requires_grad``, in either
   position, in grad mode and under ``torch.no_grad()``.  Values match torch's
   (normwise, 1e-5 at float32, 1e-12 at float64); with ``requires_grad`` in grad mode,
   the parameter's gradient matches too.
2. A tensor subclass operand (``as_subclass``): the same values.
3. A correct call on the slow path reads no operand's extents
   (``_product_operand_extents`` is not called).  The first form of this commit read
   both operands' extents there, and fd3 measured that as +1.8% to +2.3% of a 9 us call
   on the M5 (an ``nn.Parameter`` operand, 256 rows, k = 1 and 4).
4. An operand that is not a tensor (a list, a float, None, a string, an object, a numpy
   array) is refused in either position with ``matmul_operand_type_invalid``, a
   ``TypeError``, and builds nothing.
5. A dtype mismatch on the slow path is still refused with its own code, and a
   non-tensor is refused before any dtype is compared.
6. ``spmv``'s operand check, which every SpMV pays (``ops.spmv`` and ``matmul``'s
   generated matrix-vector route): a correct call never reaches its out-of-line
   refusal, and if that refusal ever returned, the check would fail closed with a code.

The judge fails a NaN or an inf explicitly; ``test_the_judge_fails_planted_errors`` is
its negative control.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

import scorch
from scorch import ops
from scorch.exceptions import (
    IndexExtentMismatchError,
    ScorchAssertionError,
    TensorTypeError,
    defect_code,
)
from scorch.stensor import STensor

ROWS, COLS, WIDTH = 9, 7, 5
SPARSE_KINDS = ["stensor_ds", "stensor_oo", "torch_csr64", "torch_csr32"]
DTYPES = [torch.float32, torch.float64]
TOL = {torch.float32: 1e-5, torch.float64: 1e-12}


class _Marked(torch.Tensor):
    """A tensor subclass with torch's default __torch_function__."""


def _dense(dtype: torch.dtype, rows: int = ROWS, cols: int = COLS) -> torch.Tensor:
    g = torch.Generator().manual_seed(11)
    d = (torch.rand(rows, cols, generator=g) < 0.4) * (
        torch.rand(rows, cols, generator=g) + 0.5
    )
    d[0, cols - 1] = 1.0  # the last column is read
    d[rows - 1, 0] = 1.0  # and the last row
    return d.to(dtype)


def _sparse(kind: str, dense: torch.Tensor):
    if kind.startswith("stensor_"):
        return STensor.from_torch(dense).to_sparse(kind[len("stensor_") :])
    csr = dense.to_sparse_csr()
    index = torch.int64 if kind == "torch_csr64" else torch.int32
    return torch.sparse_csr_tensor(
        csr.crow_indices().to(index),
        csr.col_indices().to(index),
        csr.values(),
        size=dense.shape,
    )


def _as_tensor(result) -> torch.Tensor:
    y = result if isinstance(result, torch.Tensor) else result.to_torch(False)
    y = y.detach()
    return y.to_dense() if y.layout != torch.strided else y


def _close(result, reference: torch.Tensor, dtype: torch.dtype) -> bool:
    """The normwise relative error within TOL, and every value finite: a NaN or an inf
    fails explicitly."""
    y = _as_tensor(result)
    if not bool(torch.isfinite(y).all()):
        return False
    err = float((y.double() - reference).norm() / max(float(reference.norm()), 1e-300))
    return err <= TOL[dtype]


def _rand(*shape, dtype, seed=5) -> torch.Tensor:
    return (
        torch.rand(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)
        + 0.5
    )


def test_the_judge_fails_planted_errors():
    ref = _rand(ROWS, WIDTH, dtype=torch.float64)
    for dtype in DTYPES:
        assert _close(ref.to(dtype), ref, dtype)
        for how in ("nan", "inf", "off"):
            planted = ref.to(dtype).clone()
            planted[2, 3] = float(how) if how != "off" else planted[2, 3] + 1.0
            assert not _close(planted, ref, dtype), how
            assert not _close(planted.to_sparse_csr(), ref, dtype), how


@pytest.mark.parametrize("no_grad", [False, True], ids=["grad_mode", "no_grad"])
@pytest.mark.parametrize("requires_grad", [False, True], ids=["frozen", "trained"])
@pytest.mark.parametrize("side", ["A@P", "P@A"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("kind", SPARSE_KINDS)
def test_parameter_operands(kind, dtype, side, requires_grad, no_grad):
    dense = _dense(dtype)
    a = _sparse(kind, dense)
    if side == "A@P":
        p = torch.nn.Parameter(
            _rand(COLS, WIDTH, dtype=dtype), requires_grad=requires_grad
        )
        reference = dense.double() @ p.detach().double()
    else:
        p = torch.nn.Parameter(
            _rand(WIDTH, ROWS, dtype=dtype), requires_grad=requires_grad
        )
        reference = p.detach().double() @ dense.double()

    def product():
        return scorch.matmul(a, p) if side == "A@P" else scorch.matmul(p, a)

    if no_grad:
        with torch.no_grad():
            for _ in range(3):  # the first call and calls a plan may serve
                assert _close(product(), reference, dtype)
        return
    for _ in range(3):
        y = product()
        assert _close(y, reference, dtype)
    if requires_grad:
        y = y if isinstance(y, torch.Tensor) else y.to_torch(False)
        assert y.requires_grad, "the product lost the parameter's gradient"
        g = _rand(*y.shape, dtype=dtype, seed=8)
        (y.to_dense() if y.layout != torch.strided else y).backward(g)
        expected = (
            dense.double().t() @ g.double()
            if side == "A@P"
            else g.double() @ dense.double().t()
        )
        assert p.grad is not None and _close(p.grad, expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
@pytest.mark.parametrize("kind", SPARSE_KINDS)
def test_a_tensor_subclass_operand(kind, dtype):
    dense = _dense(dtype)
    a = _sparse(kind, dense)
    x = _rand(COLS, WIDTH, dtype=dtype).as_subclass(_Marked)
    xt = _rand(WIDTH, ROWS, dtype=dtype).as_subclass(_Marked)
    for _ in range(3):
        assert _close(
            scorch.matmul(a, x),
            dense.double() @ x.as_subclass(torch.Tensor).double(),
            dtype,
        )
        assert _close(
            scorch.matmul(xt, a),
            xt.as_subclass(torch.Tensor).double() @ dense.double(),
            dtype,
        )


@pytest.mark.parametrize("kind", SPARSE_KINDS)
def test_a_correct_slow_path_call_reads_no_extents(kind, monkeypatch):
    calls: List[str] = []
    extents = ops._product_operand_extents

    def counted(operand, position):
        calls.append(position)
        return extents(operand, position)

    monkeypatch.setattr(ops, "_product_operand_extents", counted)
    dense = _dense(torch.float32)
    a = _sparse(kind, dense)
    p = torch.nn.Parameter(_rand(COLS, WIDTH, dtype=torch.float32), requires_grad=False)
    x = _rand(COLS, WIDTH, dtype=torch.float32).as_subclass(_Marked)
    xt = _rand(WIDTH, ROWS, dtype=torch.float32)
    for _ in range(3):
        scorch.matmul(a, p)
        scorch.matmul(a, x)
        if kind.startswith("stensor_"):
            scorch.matmul(
                xt, a
            )  # a dense left operand of an STensor takes the slow path too
    assert calls == [], calls
    # And the refusal still reaches it, so its message and code are the earlier ones.
    with pytest.raises(TensorTypeError):
        scorch.matmul(a, [1.0] * COLS)
    assert calls, "the refusal did not reach _product_operand_extents"


NOT_TENSORS = {
    "list": lambda: [1.0] * COLS,
    "float": lambda: 3.0,
    "None": lambda: None,
    "str": lambda: "x",
    "object": lambda: object(),
    "numpy": lambda: np.ones((COLS, WIDTH), dtype=np.float32),
}


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


@pytest.mark.parametrize("other", list(NOT_TENSORS))
@pytest.mark.parametrize("kind", SPARSE_KINDS)
def test_an_operand_that_is_not_a_tensor_is_refused_in_either_position(
    kind, other, builds
):
    a = _sparse(kind, _dense(torch.float32))
    for position, call in (
        ("second", lambda: scorch.matmul(a, NOT_TENSORS[other]())),
        ("first", lambda: scorch.matmul(NOT_TENSORS[other](), a)),
    ):
        with pytest.raises(TensorTypeError) as caught:
            call()
        assert isinstance(caught.value, TypeError)
        assert defect_code(caught.value) == "matmul_operand_type_invalid"
        assert f"matmul's {position} operand" in str(caught.value)
        assert builds == []


@pytest.mark.parametrize("kind", SPARSE_KINDS)
def test_a_dtype_mismatch_on_the_slow_path(kind):
    a = _sparse(kind, _dense(torch.float32))
    p = torch.nn.Parameter(_rand(COLS, WIDTH, dtype=torch.float64), requires_grad=False)
    with pytest.raises(RuntimeError) as mismatch:
        scorch.matmul(a, p)
    assert defect_code(mismatch.value) is not None, repr(mismatch.value)
    assert not isinstance(mismatch.value, TensorTypeError)
    # A non-tensor is refused first, whatever the other operand's dtype.
    with pytest.raises(TensorTypeError):
        scorch.matmul(np.ones((COLS, WIDTH), dtype=np.float64), a)


def test_a_correct_spmv_never_reaches_its_refusal(monkeypatch):
    calls: List[int] = []
    refuse = ops._refuse_matrix_vector_operands

    def counted(a, b):
        calls.append(1)
        return refuse(a, b)

    monkeypatch.setattr(ops, "_refuse_matrix_vector_operands", counted)
    dense = _dense(torch.float32)
    v = _rand(COLS, dtype=torch.float32)
    reference = dense.double() @ v.double()
    for fmt in ("ds", "ss"):
        a = STensor.from_torch(dense).to_sparse(fmt)
        for _ in range(3):
            assert _close(ops.spmv(a, STensor.from_torch(v)), reference, torch.float32)
            assert _close(scorch.matmul(a, v), reference, torch.float32)
    assert calls == [], calls
    with pytest.raises(IndexExtentMismatchError):
        ops.spmv(a, STensor.from_torch(v[:-1]))
    assert calls == [1]


def test_the_spmv_check_fails_closed_if_its_refusal_finds_no_mistake(monkeypatch):
    # The refusal is reached only on a mistake; were it to return, the check raises.
    monkeypatch.setattr(ops, "_refuse_matrix_vector_operands", lambda a, b: None)
    a = STensor.from_torch(_dense(torch.float32)).to_sparse("ds")
    with pytest.raises(ScorchAssertionError) as caught:
        ops.spmv(a, STensor.from_torch(_rand(COLS - 1, dtype=torch.float32)))
    assert defect_code(caught.value) == "spmv_operand_refusal_unfounded"
    # A correct call never reaches it.
    v = _rand(COLS, dtype=torch.float32)
    assert _close(
        ops.spmv(a, STensor.from_torch(v)),
        _dense(torch.float32).double() @ v.double(),
        torch.float32,
    )
