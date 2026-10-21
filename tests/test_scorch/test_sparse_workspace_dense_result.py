"""A sparse workspace drained into a result whose levels are all dense is refused by
both compilers, in both automatic arms, with ``unsupported_workspace_result_level``
(series step 16b).

The shipped lowering drains a sparse workspace into a result's trailing levels with
``CINLowerer._lower_mixed_key_drain``, which stores a dense leaf into the value vector a
sparse result grows.  It admitted a result whose levels are all dense, which has no
such vector: its values are one array of its full size.  The kernel called ``size``,
``resize`` and ``scorch_vector_set`` on that array, and its C++ did not build.  The
lowering now refuses the program before emitting anything, with the code it gives a
sparse workspace drained into a dense level.  The new compiler (LoopIR), which lowers
no Where program, refuses this one with the same code.  Supporting the drain into a
dense result is later work.

The program is ``C(i, j) = w(j)`` where ``w(j) += A(i, k) * B(k, j)``, with C stored
``dd``, A ``ds`` or ``ss`` (row 1 empty) and B ``ds``.  Two controls show that the
refusal is this drain's alone: with a dense workspace the same program is computed,
and a sparse workspace drained into an ``sd`` result, the kind of result the drain is
for, still is.  Two more programs bound the new compiler's refusal, so that it
refuses with this code only what the shipped lowering refuses with it.  A Where below
no loop is computed by the shipped lowering, through a coordinate intermediate and a
conversion, and keeps the new compiler's general refusal, ``unsupported_statement``.
A consumer inside a reduction loop over ``k`` is refused by the shipped lowering with
``unsupported_workspace_inside_reduction`` and keeps that general refusal too.
Values are compared with a float64 reference to a relative 1e-12.
"""

from __future__ import annotations

from typing import Any, Tuple

import pytest
import torch

from scorch import STensor, ops  # type: ignore[import-untyped]
from scorch.compiler.cin import (  # type: ignore[import-untyped]
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
    Workspace,
)
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)
from scorch.compiler.loopir.lower_cin import (  # type: ignore[import-untyped]
    LoopIRLoweringError,
)
from scorch.compiler.loopir.pipeline import (  # type: ignore[import-untyped]
    execute_cin_via_loopir,
)
from scorch.exceptions import defect_code  # type: ignore[import-untyped]
from tests.test_scorch.test_sparse_assembly_strategy import auto_options

REFUSAL = "unsupported_workspace_result_level"
F64 = torch.float64
N, K, M = 7, 6, 5


def _set_arm(monkeypatch: pytest.MonkeyPatch, arm: int) -> None:
    monkeypatch.setenv("SCORCH_REGBLOCK", str(arm))
    assert CompileOptions.from_environment().scheduler.regblock_enabled == bool(arm)


def _operands(a_format: str) -> Tuple[STensor, STensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(16)
    a = torch.rand(N, K, generator=generator, dtype=F64)
    a[a < 0.5] = 0.0
    a[1] = 0.0
    b = torch.rand(K, M, generator=generator, dtype=F64)
    b[b < 0.4] = 0.0
    a_tensor = STensor.from_torch(a).to_sparse(a_format)
    b_tensor = STensor.from_torch(b).to_sparse("ds")
    return a_tensor, b_tensor, a @ b


def _program(a_format: str, result_format: str, dense_workspace: bool) -> Any:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt=result_format, shape=(N, M), dtype=F64)
    a = TensorVar("A", fmt=a_format, shape=(N, K), dtype=F64)
    b = TensorVar("B", fmt="ds", shape=(K, M), dtype=F64)
    w = Workspace(name="w", dim=1, dtype=F64, dense=dense_workspace)
    return ForAll(
        i,
        Where(
            producer=ForAll(
                k, ForAll(j, TensorAssign(w[j], a[i, k] * b[k, j], op=Operation.ADD))
            ),
            consumer=ForAll(j, TensorAssign(c[i, j], w[j])),
        ),
    )


def _program_below_no_loop() -> Any:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dd", shape=(N, M), dtype=F64)
    a = TensorVar("A", fmt="ds", shape=(N, K), dtype=F64)
    b = TensorVar("B", fmt="ds", shape=(K, M), dtype=F64)
    w = Workspace(name="w", dim=2, dtype=F64, dense=False)
    return Where(
        producer=ForAll(
            i,
            ForAll(
                k,
                ForAll(j, TensorAssign(w[i, j], a[i, k] * b[k, j], op=Operation.ADD)),
            ),
        ),
        consumer=ForAll(i, ForAll(j, TensorAssign(c[i, j], w[i, j]))),
    )


def _program_inside_a_reduction() -> Any:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dd", shape=(N, M), dtype=F64)
    a = TensorVar("A", fmt="ds", shape=(N, K), dtype=F64)
    b = TensorVar("B", fmt="ds", shape=(K, M), dtype=F64)
    w = Workspace(name="w", dim=1, dtype=F64, dense=False)
    return ForAll(
        i,
        ForAll(
            k,
            Where(
                producer=ForAll(
                    j, TensorAssign(w[j], a[i, k] * b[k, j], op=Operation.ADD)
                ),
                consumer=ForAll(j, TensorAssign(c[i, j], w[j], op=Operation.ADD)),
            ),
        ),
    )


def _dense(result: Any) -> torch.Tensor:
    values = result.to_torch()
    return values if values.layout == torch.strided else values.to_dense()


def _shipped_refusal(
    monkeypatch: pytest.MonkeyPatch, program: Any, a: STensor, b: STensor
) -> Any:
    def no_build(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a kernel was built for a program the lowering refuses")

    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", no_build)
    with pytest.raises(Exception) as caught:
        ops.lower_and_exec_cin(program, (N, M), a, b)
    return caught.value


def _new_compiler_refusal(program: Any, a: STensor, b: STensor, arm: int) -> Any:
    with pytest.raises(LoopIRLoweringError) as caught:
        execute_cin_via_loopir(
            program, (N, M), a, b, compile_options=auto_options(bool(arm), jit=True)
        )
    return caught.value


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("a_format", ["ds", "ss"])
def test_the_shipped_compiler_refuses_the_drain(
    monkeypatch: pytest.MonkeyPatch, a_format: str, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    a, b, _ = _operands(a_format)
    error = _shipped_refusal(monkeypatch, _program(a_format, "dd", False), a, b)
    assert defect_code(error) == REFUSAL, error


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("a_format", ["ds", "ss"])
def test_the_new_compiler_refuses_it_with_the_same_code(
    a_format: str, arm: int
) -> None:
    a, b, _ = _operands(a_format)
    error = _new_compiler_refusal(_program(a_format, "dd", False), a, b, arm)
    assert error.defect.code == REFUSAL, error


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("a_format", ["ds", "ss"])
def test_the_program_with_a_dense_workspace_is_computed(
    monkeypatch: pytest.MonkeyPatch, a_format: str, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    a, b, want = _operands(a_format)
    got = ops.lower_and_exec_cin(_program(a_format, "dd", True), (N, M), a, b)
    assert torch.allclose(_dense(got), want, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("arm", [0, 1])
def test_a_sparse_workspace_into_an_sd_result_is_still_computed(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    a, b, want = _operands("ds")
    got = ops.lower_and_exec_cin(_program("ds", "sd", False), (N, M), a, b)
    assert torch.allclose(_dense(got), want, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("arm", [0, 1])
def test_the_shipped_compiler_computes_a_where_below_no_loop(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    a, b, want = _operands("ds")
    got = ops.lower_and_exec_cin(_program_below_no_loop(), (N, M), a, b)
    assert torch.allclose(_dense(got), want, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("arm", [0, 1])
def test_the_new_compiler_keeps_the_general_refusal_below_no_loop(arm: int) -> None:
    a, b, _ = _operands("ds")
    error = _new_compiler_refusal(_program_below_no_loop(), a, b, arm)
    assert error.defect.code == "unsupported_statement", error


@pytest.mark.parametrize("arm", [0, 1])
def test_the_shipped_compiler_refuses_a_drain_inside_a_reduction(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    a, b, _ = _operands("ds")
    error = _shipped_refusal(monkeypatch, _program_inside_a_reduction(), a, b)
    assert defect_code(error) == "unsupported_workspace_inside_reduction", error


@pytest.mark.parametrize("arm", [0, 1])
def test_the_new_compiler_keeps_the_general_refusal_inside_a_reduction(
    arm: int,
) -> None:
    a, b, _ = _operands("ds")
    error = _new_compiler_refusal(_program_inside_a_reduction(), a, b, arm)
    assert error.defect.code == "unsupported_statement", error
