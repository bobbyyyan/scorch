"""A rank-0 tensor is a tensor with a format, and both compilers refuse it by code.

``TensorVar(fmt="")`` tested the format for truth, so the empty string -- which
``parse_format`` reads as the rank-0 ``TensorFormat('')`` -- was dropped and the
scalar was left with no format and no mode order.  With a requested schedule the new
compiler then failed in the scheduler with ``TypeError: 'NoneType' object is not
iterable``.  Declared with a ``TensorFormat()`` object instead, the same scalar result
went through the shipped lowering, which emitted a kernel that reads an undeclared
``y0_size``; a scalar operand raised ``IndexError``.

Neither compiler lowers a rank-0 tensor.  Each now says so with a structured code.
"""

from dataclasses import replace

import pytest
import torch

from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import UnsupportedFeature
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir, legacy_generated_cpp
from scorch.compiler.scheduler import Schedule
from scorch.format import TensorFormat
from tests.test_scorch.level_array_support import emitted_einsum_cpp

F64 = torch.float64
SPELLINGS = pytest.mark.parametrize(
    "scalar_format",
    ("", [], TensorFormat()),
    ids=("empty-string", "empty-list", "object"),
)


def scalar_result(scalar_format):
    """``y[] = sum_ab A[a, b]``."""

    a, b = IndexVar("a"), IndexVar("b")
    operand = TensorVar("A", fmt="ss", dtype=F64)
    result = TensorVar("y", fmt=scalar_format, dtype=F64)
    assign = TensorAssign(result[()], operand[a, b], op=Operation.ADD)
    return ForAll(a, ForAll(b, assign)), (), (((3, 4), F64),)


def scalar_operand(scalar_format):
    """``y[a] = x[a] * s[]``."""

    a = IndexVar("a")
    vector = TensorVar("x", fmt="d", dtype=F64)
    scale = TensorVar("s", fmt=scalar_format, dtype=F64)
    result = TensorVar("y", fmt="d", dtype=F64)
    assign = TensorAssign(result[a], vector[a] * scale[()])
    return ForAll(a, assign), (3,), (((3,), F64), ((), F64))


PROGRAMS = pytest.mark.parametrize(
    ("program", "named"), ((scalar_result, "'y'"), (scalar_operand, "'s'"))
)


@SPELLINGS
def test_an_empty_format_is_the_rank_zero_format(scalar_format):
    scalar = TensorVar("y", fmt=scalar_format, dtype=F64)

    assert scalar.format is not None
    assert scalar.format.get_order() == 0
    assert scalar.mode_order == []


def test_no_format_is_still_no_format():
    unformatted = TensorVar("y", dtype=F64)

    assert unformatted.format is None
    assert unformatted.mode_order is None


@pytest.mark.parametrize("scheduled", (False, True), ids=("unscheduled", "scheduled"))
@pytest.mark.parametrize("register_blocking", (False, True), ids=("off", "on"))
@SPELLINGS
@PROGRAMS
def test_new_compiler_refuses_a_scalar_tensor_as_unsupported(
    program, named, scalar_format, register_blocking, scheduled
):
    statement, result_shape, bindings = program(scalar_format)
    options = CompileOptions.from_environment().with_regblock_enabled(register_blocking)
    if scheduled:
        options = replace(options, requested_schedule=Schedule())

    with pytest.raises(LoopIRLoweringError) as refusal:
        compile_cin_via_loopir(
            statement, result_shape, bindings, compile_options=options
        )

    assert refusal.value.defect.code == "unsupported_format"
    assert f"tensor {named} is rank 0" in refusal.value.defect.message


@SPELLINGS
@PROGRAMS
def test_shipped_lowering_refuses_a_scalar_tensor_by_code(
    program, named, scalar_format
):
    statement, result_shape, bindings = program(scalar_format)

    with pytest.raises(UnsupportedFeature) as refusal:
        legacy_generated_cpp(statement, result_shape, bindings)

    (diagnostic,) = refusal.value.diagnostics
    assert diagnostic.code == "unsupported_scalar_tensor"
    assert diagnostic.stage == "legacy_cin_lowering"
    assert f"tensor {named}" in str(refusal.value)


@pytest.mark.parametrize(
    ("expr", "formats", "result_format", "marker"),
    [
        ("ab->a", ["ss"], "d", "B_values[pB0] += A_val[pA1];"),
        ("ab,bc->ac", ["ds", "ds"], "ds", "wksp.insert_unchecked({c}, "),
        ("ab,cb->ac", ["ds", "dd"], "dd", "double wksp[kTile_c] = {};"),
    ],
)
def test_programs_with_accumulators_still_lower(
    monkeypatch, expr, formats, result_format, marker
):
    """The refusal is for a declared rank-0 tensor, not for a kernel's scalar or
    workspace accumulator."""

    extents = {name: 3 for name in expr if name.isalpha()}

    source = emitted_einsum_cpp(
        monkeypatch, expr, formats, extents, format=result_format
    )

    assert "Tensor evaluate(" in source
    assert marker in source
