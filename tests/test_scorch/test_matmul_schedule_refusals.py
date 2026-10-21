"""A schedule for a product no compiler path can schedule is refused with a code.

Two causes, one class and one code each:
- two dense operands, torch tensors or ``STensor`` operands, which ``torch.matmul``
  multiplies: ``ScheduleWithoutSparseOperandError``, code
  ``schedule_without_sparse_operand``;
- a 2-D times 1-D product with a sparse matrix, which goes to the SpMV path,
  reached from ``matmul`` or by calling ``ops.spmv`` directly:
  ``ScheduleNotThreadedThroughSpMVError``, code
  ``schedule_not_threaded_through_spmv``.
A schedule comes three ways: the ``schedule=`` keyword, a ``schedule_force``
context, or ``_compile_options`` with ``requested_schedule`` set.  For every
product, entry point and way, the refusal is the cause's class with its code,
before any kernel build is requested.  Both classes are ``NotImplementedError``
(``matmul``'s refusals before they had a code) and the SpMV one is also a
``CompileOptionsError`` (``ops.spmv``'s), so either ``except`` clause still catches
them.  Without a schedule the same operands are multiplied (the control).
"""

import functools
import pickle

import pytest
import torch

import scorch
from scorch import ops, stensor, utils
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import (
    CompileOptionsError,
    ScheduleNotThreadedThroughSpMVError,
)
from scorch.compiler.scheduler import Schedule, schedule_force
from scorch.exceptions import (
    ScheduleWithoutSparseOperandError,
    ScorchError,
    UnsupportedScheduleError,
)
from scorch.stensor import STensor
from tests.test_scorch.level_array_support import stored

DENSE_TEXT = "Explicit compiler schedules currently require a sparse operand"
SPMV_TEXT = "Explicit schedules are not yet threaded through the SpMV compiler"


def _dense(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=generator, dtype=torch.float64)


def _sparse_matrix():
    generator = torch.Generator().manual_seed(7)
    keep = torch.rand((6, 5), generator=generator) < 0.5
    return torch.rand((6, 5), generator=generator, dtype=torch.float64) * keep


# name -> (the operands, the error, the float64 reference product)
DENSE_PRODUCTS = {
    "torch_dense_pair": (
        lambda: (_dense((4, 5), 1), _dense((5, 3), 2)),
        lambda: _dense((4, 5), 1) @ _dense((5, 3), 2),
    ),
    "torch_dense_matrix_vector": (
        lambda: (_dense((4, 5), 1), _dense((5,), 3)),
        lambda: _dense((4, 5), 1) @ _dense((5,), 3),
    ),
    "stensor_dense_pair": (
        lambda: (
            stored(_dense((4, 5), 1), "dd", "A"),
            stored(_dense((5, 3), 2), "dd", "B"),
        ),
        lambda: _dense((4, 5), 1) @ _dense((5, 3), 2),
    ),
}
SPMV_PRODUCTS = {
    "ds_times_torch_vector": lambda: (
        stored(_sparse_matrix(), "ds", "A"),
        _dense((5,), 3),
    ),
    "torch_csr_times_torch_vector": lambda: (
        _sparse_matrix().to_sparse_csr(),
        _dense((5,), 3),
    ),
    "oo_times_stensor_vector": lambda: (
        stored(_sparse_matrix(), "oo", "A"),
        stored(_dense((5,), 3), "d", "x"),
    ),
}
SPELLINGS = ("keyword", "schedule_force", "compile_options")
ENTRIES = ("matmul", "ops.spmv")


def _as_stensor(operand):
    if isinstance(operand, STensor):
        return operand
    if operand.layout == torch.sparse_csr:
        return STensor.from_torch(operand)
    return stored(operand, "d" * operand.dim(), "x")


def _multiply(entry, spelling, a, b):
    if entry == "ops.spmv":
        product = functools.partial(ops.spmv, _as_stensor(a), _as_stensor(b), "d")
    else:
        product = functools.partial(scorch.matmul, a, b)
    schedule = Schedule(tag=f"refused-{spelling}")
    if spelling == "keyword":
        return product(schedule=schedule)
    if spelling == "schedule_force":
        with schedule_force(schedule):
            return product()
    options = CompileOptions.from_environment(requested_schedule=schedule)
    assert options.requested_schedule is schedule
    return product(_compile_options=options)


@pytest.fixture
def builds(monkeypatch):
    """Every kernel build request, recorded and refused: the list must stay empty."""

    requested = []

    def refuse(*args, **kwargs):
        requested.append(args)
        raise AssertionError("a kernel build was requested before the refusal")

    entries = (
        "_prepare_generated_kernel_build",
        "_prepare_jit_build",
        "_load_kernel",
        "_load_validated_prepared_kernel",
    )
    patched = 0
    for module in (ops, utils, stensor):
        for name in entries:
            if hasattr(module, name):
                monkeypatch.setattr(module, name, refuse)
                patched += 1
    assert patched >= 4, "the kernel build entry points were not found"
    return requested


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("product", sorted(DENSE_PRODUCTS))
def test_a_dense_product_refuses_a_schedule(builds, product, spelling):
    a, b = DENSE_PRODUCTS[product][0]()
    with pytest.raises(UnsupportedScheduleError) as raised:
        _multiply("matmul", spelling, a, b)
    error = raised.value
    assert type(error) is ScheduleWithoutSparseOperandError
    assert error.code == "schedule_without_sparse_operand"
    assert isinstance(error, NotImplementedError)
    assert isinstance(error, ScorchError)
    assert str(error) == DENSE_TEXT
    assert builds == []


@pytest.mark.parametrize("spelling", SPELLINGS)
@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("product", sorted(SPMV_PRODUCTS))
def test_a_matrix_vector_product_refuses_a_schedule(builds, product, entry, spelling):
    a, b = SPMV_PRODUCTS[product]()
    with pytest.raises(UnsupportedScheduleError) as raised:
        _multiply(entry, spelling, a, b)
    error = raised.value
    assert type(error) is ScheduleNotThreadedThroughSpMVError
    assert error.code == "schedule_not_threaded_through_spmv"
    assert isinstance(error, NotImplementedError)
    assert isinstance(error, ScorchError)
    assert isinstance(error, CompileOptionsError)
    assert [(d.code, d.field) for d in error.diagnostics] == [
        ("schedule_not_threaded_through_spmv", "requested_schedule")
    ]
    assert SPMV_TEXT in str(error)
    assert builds == []


def test_each_cause_has_one_code_everywhere(builds):
    codes = {}
    cases = [("matmul", name, DENSE_PRODUCTS[name][0]) for name in DENSE_PRODUCTS]
    cases += [
        (entry, name, SPMV_PRODUCTS[name])
        for name in SPMV_PRODUCTS
        for entry in ENTRIES
    ]
    for entry, name, operands in cases:
        cause = "dense" if name in DENSE_PRODUCTS else "spmv"
        for spelling in SPELLINGS:
            a, b = operands()
            try:
                _multiply(entry, spelling, a, b)
            except UnsupportedScheduleError as error:
                codes.setdefault(cause, set()).add((type(error), error.code))
            else:
                raise AssertionError(f"{entry} {name} {spelling} was not refused")
    assert codes == {
        "dense": {
            (ScheduleWithoutSparseOperandError, "schedule_without_sparse_operand")
        },
        "spmv": {
            (ScheduleNotThreadedThroughSpMVError, "schedule_not_threaded_through_spmv")
        },
    }
    assert builds == []


@pytest.mark.parametrize(
    "error",
    [ScheduleWithoutSparseOperandError(), ScheduleNotThreadedThroughSpMVError()],
)
def test_a_refusal_survives_pickling(error):
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is type(error)
    assert copy.code == error.code
    assert str(copy) == str(error)


@pytest.mark.parametrize("product", sorted(DENSE_PRODUCTS))
def test_without_a_schedule_a_dense_product_runs(product):
    operands, reference = DENSE_PRODUCTS[product]
    a, b = operands()
    result = scorch.matmul(a, b)
    if isinstance(result, STensor):
        result = result.to_torch(in_place=False)
    torch.testing.assert_close(result, reference(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("product", sorted(SPMV_PRODUCTS))
def test_without_a_schedule_a_matrix_vector_product_runs(product, entry):
    a, b = SPMV_PRODUCTS[product]()
    if entry == "ops.spmv":
        result = ops.spmv(_as_stensor(a), _as_stensor(b), "d")
    else:
        result = scorch.matmul(a, b)
    if isinstance(result, STensor):
        result = result.to_torch(in_place=False)
    elif result.layout != torch.strided:
        result = result.to_dense()
    assert result.dtype == torch.float64
    reference = _sparse_matrix() @ _dense((5,), 3)
    torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
