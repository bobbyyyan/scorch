"""Products and sums of operands with different value dtypes.

``A + B`` and ``matmul(A, x)`` lower their own programs (neither goes through
``einsum``), and both declared the result with the *left* operand's dtype.  The kernel
computed each element in C++ in the wider type and then stored it in the left operand's:
an int64 ``A`` plus a float64 ``B`` returned int64, every sum truncated toward zero; an
int64 matrix times a float64 vector accumulated each row in an ``int64_t`` scalar;
float32 with float64 returned float32; int32 with int64 wrapped past 2**31.  Nothing was
refused and nothing crashed.

A sum now has the operands' common dtype by PyTorch's promotion rule
(``layout.common_value_dtype``), as ``torch`` gives for ``a + b``; int8 with uint8
promotes to int16, which Scorch does not store, so that pair is refused.

A product does not promote: ``torch.matmul``, ``mv``, ``mm``, ``@`` and ``einsum`` raise
``RuntimeError`` for operands of different dtypes (torch 2.13), and Bobby decided on
2026-10-12 that Scorch's do the same.  Every product entry point raises
``OperandDtypeMismatchError`` (a ``RuntimeError`` and a ``TensorValidationError``, code
``operand_dtype_mismatch``) before it looks up or builds any kernel; the tests run
each such call with every build, prebuilt-kernel and plan function replaced by one that
fails.

Operands are built and results read as level arrays in PyTorch.
"""

import re

import pytest
import torch

import scorch
from scorch import ops, stensor
from scorch import utils
from scorch.exceptions import OperandDtypeMismatchError, TensorValidationError
from scorch.layout import (
    SUPPORTED_VALUE_DTYPES,
    common_value_dtype,
    require_same_value_dtype,
)
from tests.test_scorch.kernel_cache_support import empty_kernel_caches
from tests.test_scorch.level_array_support import logical_dense, stored

C_TYPE = {
    torch.float32: "float",
    torch.float64: "double",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
}
# (left dtype, right dtype): the first three stored a narrower type than torch's; the
# last two already stored the common type (the wider operand is on the left).
MIXED = [
    (torch.int64, torch.float64),
    (torch.float32, torch.float64),
    (torch.int32, torch.int64),
    (torch.float64, torch.int64),
    (torch.int64, torch.int32),
]
ADDITION_LAYOUTS = [("ss", "ss"), ("dd", "ss"), ("oo", "dd")]
SHAPE = (5, 4)


class _Emitted(Exception):
    """Raised in place of building the kernel."""


def operand(shape, dtype, seed):
    """About half the entries zero.  Float values have a fractional part and int64
    values exceed 2**31, so a result stored in a narrower type is far from right."""

    generator = torch.Generator().manual_seed(seed)
    keep = torch.rand(shape, generator=generator, dtype=torch.float64) < 0.5
    if dtype.is_floating_point:
        values = 0.25 + torch.rand(shape, generator=generator, dtype=torch.float64)
    elif dtype == torch.int64:
        values = torch.randint(1, 100, shape, generator=generator) + 3_000_000_000
    else:
        values = torch.randint(1, 100, shape, generator=generator)
    return (values * keep).to(dtype)


def emitted(monkeypatch, call):
    """The kernel ``call`` emits; nothing is built."""

    captured = []

    def stop_before_native(prepared):
        captured.append(prepared.request.cpp_sources[-1])
        raise _Emitted

    # A kernel an earlier test kept for this call would serve it without reaching the stop.
    empty_kernel_caches(monkeypatch)
    monkeypatch.setattr(stensor, "_load_validated_prepared_kernel", stop_before_native)
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", stop_before_native)
    with pytest.raises(_Emitted):
        call()
    return captured[0][captured[0].index("evaluate(") :]


def stored_value_type(source, result):
    """The C++ element type of ``result``'s value storage in ``source``."""

    matched = re.search(
        rf"std::vector<(\w+)> {result}_values;|"
        rf"{result}_values_torch\.data_ptr<(\w+)>\(\)",
        source,
    )
    assert matched, f"no value storage for {result}"
    return matched.group(1) or matched.group(2)


def assert_equals_torch(result, expected):
    """The dtype ``torch`` gives, and its values: integers exactly, float64 to 1e-12,
    float32 to 1e-5."""

    assert result.dtype == expected.dtype
    got = logical_dense(result)
    if not expected.dtype.is_floating_point:
        assert torch.equal(got, expected.to(torch.float64))
        return
    tolerance = 1e-12 if expected.dtype == torch.float64 else 1e-5
    assert torch.allclose(got, expected.to(torch.float64), rtol=tolerance, atol=0.0)


def test_common_dtype_is_torch_promotion():
    for dtype in C_TYPE:
        assert common_value_dtype(dtype, dtype, operation="addition") == dtype
    for left, right in MIXED:
        assert common_value_dtype(
            left, right, operation="addition"
        ) == torch.promote_types(left, right)
    assert (
        common_value_dtype(torch.int64, torch.float32, operation="addition")
        == torch.float32
    )


@pytest.mark.parametrize(("left", "right"), ADDITION_LAYOUTS)
def test_int8_with_uint8_is_refused(left, right):
    a = stored(operand(SHAPE, torch.int8, 3), left, "A")
    b = stored(operand(SHAPE, torch.uint8, 4), right, "B")
    with pytest.raises(TensorValidationError, match="needs torch.int16"):
        a + b
    with pytest.raises(TensorValidationError, match="needs torch.int16"):
        b + a


@pytest.mark.parametrize(("left_dtype", "right_dtype"), MIXED)
@pytest.mark.parametrize(("left", "right"), ADDITION_LAYOUTS)
def test_addition_stores_the_common_type(
    monkeypatch, left, right, left_dtype, right_dtype
):
    a = stored(operand(SHAPE, left_dtype, 3), left, "A")
    b = stored(operand(SHAPE, right_dtype, 4), right, "B")

    source = emitted(monkeypatch, lambda: a + b)

    common = torch.promote_types(left_dtype, right_dtype)
    assert stored_value_type(source, "A") == C_TYPE[common]
    assert f"B_val = B_values.data_ptr<{C_TYPE[left_dtype]}>();" in source
    assert f"C_val = C_values.data_ptr<{C_TYPE[right_dtype]}>();" in source


@pytest.mark.parametrize(("left_dtype", "right_dtype"), MIXED)
@pytest.mark.parametrize(("left", "right"), ADDITION_LAYOUTS)
def test_addition_of_mixed_dtypes_equals_torch(left, right, left_dtype, right_dtype):
    a = operand(SHAPE, left_dtype, 7)
    b = operand(SHAPE, right_dtype, 8)

    result = stored(a, left, "A") + stored(b, right, "B")

    assert_equals_torch(result, a + b)


# Every ordered pair of different dtypes Scorch stores.
PAIRS = [
    (left, right)
    for left in SUPPORTED_VALUE_DTYPES
    for right in SUPPORTED_VALUE_DTYPES
    if left != right
]


class _LookedUp(Exception):
    """Raised by a build, prebuilt-kernel or plan function a mixed product reached."""


def _fail(*args, **kwargs):
    raise _LookedUp


# (name, scorch call, torch call, left shape, right shape).  The calls take the two
# dense operands; each scorch call stores them in its own layout first.
PRODUCTS = [
    (
        "sparse matrix x sparse-library vector",
        lambda a, x: scorch.matmul(stored(a, "ds", "A"), stored(x, "d", "x")),
        torch.matmul,
        (5, 4),
        (4,),
    ),
    (
        "sparse matrix x torch vector",
        lambda a, x: scorch.matmul(stored(a, "ds", "A"), x),
        torch.matmul,
        (5, 4),
        (4,),
    ),
    (
        "coordinate matrix x torch vector",
        lambda a, x: scorch.matmul(stored(a, "oo", "A"), x),
        torch.matmul,
        (5, 4),
        (4,),
    ),
    (
        "sparse matrix x torch matrix",
        lambda a, b: scorch.matmul(stored(a, "ds", "A"), b),
        torch.matmul,
        (5, 4),
        (4, 3),
    ),
    (
        "sparse matrix x sparse matrix",
        lambda a, b: scorch.matmul(stored(a, "ds", "A"), stored(b, "ds", "B")),
        torch.matmul,
        (5, 4),
        (4, 3),
    ),
    (
        "torch CSR matrix x torch matrix",
        lambda a, b: scorch.matmul(a.to_sparse_csr(), b),
        torch.matmul,
        (5, 4),
        (4, 3),
    ),
    (
        "torch matrix x sparse matrix",
        lambda x, b: scorch.matmul(x, stored(b, "ds", "B")),
        torch.matmul,
        (3, 5),
        (5, 4),
    ),
    (
        "torch vector x sparse matrix",
        lambda x, b: scorch.matmul(x, stored(b, "ds", "B")),
        torch.matmul,
        (5,),
        (5, 4),
    ),
    (
        "einsum ij,j->i",
        lambda a, x: scorch.einsum(
            "ij,j->i", stored(a, "ds", "A"), stored(x, "d", "x"), format="d"
        ),
        lambda a, x: torch.einsum("ij,j->i", a, x),
        (5, 4),
        (4,),
    ),
    (
        "einsum ij,jk->ik",
        lambda a, b: scorch.einsum(
            "ij,jk->ik", stored(a, "ds", "A"), stored(b, "dd", "B"), format="dd"
        ),
        lambda a, b: torch.einsum("ij,jk->ik", a, b),
        (5, 4),
        (4, 3),
    ),
    (
        "ops.spmv",
        lambda a, x: ops.spmv(stored(a, "ds", "A"), stored(x, "d", "x"), "d"),
        torch.mv,
        (5, 4),
        (4,),
    ),
]


def test_one_dtype_is_returned_and_two_are_refused():
    for dtype in SUPPORTED_VALUE_DTYPES:
        assert require_same_value_dtype(dtype, dtype, operation="matmul") == dtype
    with pytest.raises(OperandDtypeMismatchError, match="same value dtype") as raised:
        require_same_value_dtype(torch.float32, torch.float64, operation="matmul")
    assert isinstance(raised.value, RuntimeError)
    assert isinstance(raised.value, TensorValidationError)
    assert raised.value.code == "operand_dtype_mismatch"


@pytest.mark.parametrize(("left_dtype", "right_dtype"), PAIRS)
@pytest.mark.parametrize(
    ("scorch_call", "torch_call", "left_shape", "right_shape"),
    [entry[1:] for entry in PRODUCTS],
    ids=[entry[0] for entry in PRODUCTS],
)
def test_a_product_of_mixed_dtypes_raises_as_torch_does(
    monkeypatch,
    scorch_call,
    torch_call,
    left_shape,
    right_shape,
    left_dtype,
    right_dtype,
):
    left = operand(left_shape, left_dtype, 11)
    right = operand(right_shape, right_dtype, 12)
    with pytest.raises(Exception) as from_torch:
        torch_call(left, right)
    for module, name in [
        (utils, "_prepare_jit_build"),
        (utils, "_load_validated_prepared_kernel"),
        (ops, "_prepare_jit_build"),
        (ops, "_load_validated_prepared_kernel"),
        (stensor, "_prepare_jit_build"),
        (stensor, "_load_validated_prepared_kernel"),
        (ops, "resolve_prebuilt_matmul"),
        (ops, "_run_torch_operand_plan"),
        (ops._autograd, "serve"),
    ]:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _fail)

    with pytest.raises(type(from_torch.value)) as from_scorch:
        scorch_call(left, right)

    assert isinstance(from_scorch.value, OperandDtypeMismatchError)
    assert from_scorch.value.code == "operand_dtype_mismatch"


def test_dense_operands_of_mixed_dtypes_raise_torchs_own_error():
    a = operand((5, 4), torch.float32, 13)
    b = operand((4, 3), torch.float64, 14)
    with pytest.raises(RuntimeError, match="same dtype"):
        scorch.matmul(a, b)


def dense_of(result):
    if isinstance(result, torch.Tensor):
        return (result.to_dense() if result.layout != torch.strided else result).to(
            torch.float64
        )
    return logical_dense(result)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("scorch_call", "torch_call", "left_shape", "right_shape"),
    [entry[1:] for entry in PRODUCTS],
    ids=[entry[0] for entry in PRODUCTS],
)
def test_a_product_of_one_dtype_equals_torch(
    scorch_call, torch_call, left_shape, right_shape, dtype
):
    left = operand(left_shape, dtype, 15)
    right = operand(right_shape, dtype, 16)

    got = dense_of(scorch_call(left, right))

    expected = torch_call(left.to(torch.float64), right.to(torch.float64))
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    assert torch.allclose(got, expected, rtol=tolerance, atol=0.0)
