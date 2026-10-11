"""Conversions a scheduled call performs lower without the call's schedule.

``scorch.einsum(..., schedule=...)`` and ``scorch.matmul(..., schedule=...)``
may convert an operand or the result to another mode order or format before or
after the scheduled kernel runs.  Each conversion is a separate kernel with no
schedule of its own.  Plain CIN lowering refuses to ignore a requested schedule,
so a conversion handed the caller's options failed with
``unsupported_requested_schedule``: ``einsum('ijk->i', A, format='s',
schedule=Schedule())`` on a dense ``A`` raised before its kernel was built.
864745d did not refuse it -- the guard arrived with the refactor merge -- so
this was lost capability.  The conversions now lower with the schedule removed
and under their own timing owner, as the LoopIR relayout prerequisite does.
"""

from dataclasses import replace

import pytest
import torch

import scorch
from scorch.compiler.compilation_context import CompilationContext
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.scheduler import Schedule
from scorch.ops import _conversion_compile_inputs
from scorch.stensor import STensor

ATOL = RTOL = 1e-3


def test_the_conversion_inputs_drop_only_the_requested_schedule():
    plain = CompileOptions.from_environment()
    context = CompilationContext(compile_options=plain)
    options, owner = _conversion_compile_inputs(plain, context)
    assert options is plain and owner is context
    options, owner = _conversion_compile_inputs(plain, None)
    assert options is plain and owner is None

    scheduled = CompileOptions.from_environment(requested_schedule=Schedule())
    caller_context = CompilationContext(compile_options=scheduled)
    options, owner = _conversion_compile_inputs(scheduled, caller_context)
    assert options.requested_schedule is None
    assert options == replace(scheduled, requested_schedule=None)
    assert owner is not caller_context
    assert owner.compile_options is options


class _Stop(Exception):
    pass


def _dense_operand(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator) + 0.5
    return values * (torch.rand(shape, generator=generator) < 0.5)


def test_a_scheduled_einsum_hands_its_conversion_schedule_free_options(monkeypatch):
    """No kernel is built: the conversion's own inputs are captured, then stopped."""

    observed = []

    def capture(self, mode_order, **kwargs):
        observed.append((kwargs["_compile_options"], kwargs["_compilation_context"]))
        raise _Stop

    monkeypatch.setattr(STensor, "change_mode_order", capture)
    operand = STensor.from_torch(_dense_operand((3, 4, 5), seed=0), "A")
    with pytest.raises(_Stop):
        scorch.einsum("ijk->i", operand, format="s", schedule=Schedule())
    assert len(observed) == 1
    options, owner = observed[0]
    assert options.requested_schedule is None
    assert owner.compile_options is options


@pytest.mark.parametrize(
    "spec,operand_fmt,result_fmt",
    [
        ("ijk->i", "ddd", "s"),
        ("ijk->ij", "ddd", "ss"),
        ("ijk->ik", "ddd", "ds"),
        ("ijk->k", "sss", "s"),
        ("ij->ji", "ss", "ss"),
    ],
)
def test_a_scheduled_einsum_matches_the_unscheduled_one(spec, operand_fmt, result_fmt):
    operand_shape = (3, 4, 5)[: len(spec.split("->")[0])]
    dense = _dense_operand(operand_shape, seed=7)
    expected = torch.einsum(spec, dense.to(torch.float64))

    def operand():
        tensor = STensor.from_torch(dense.clone(), "A")
        if set(operand_fmt) != {"d"}:
            tensor.to_sparse(operand_fmt)
        return tensor

    unscheduled = scorch.einsum(spec, operand(), format=result_fmt)
    scheduled = scorch.einsum(spec, operand(), format=result_fmt, schedule=Schedule())
    for result in (unscheduled, scheduled):
        assert str(result.format) == ",".join(result_fmt)
        torch.testing.assert_close(
            result.to_torch(in_place=False).to(torch.float64),
            expected,
            atol=ATOL,
            rtol=RTOL,
        )
