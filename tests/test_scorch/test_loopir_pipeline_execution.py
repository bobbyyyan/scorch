"""End-to-end LoopIR dense vertical slice: compile, execute, and compare.

Curated differential execution for the migrated families: the LoopIR path's
kernel must agree bitwise with the untouched legacy path (their generated
sources are byte-identical, so they share one kernel artifact), match the
PyTorch reference within the repository tolerance convention, and match the
production LoopIR oracle.  This module is the only place that runs both
pipelines on real tensors; nothing enables shadow execution globally.
"""

import copy

import pytest
import torch

from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compilation_context import (
    CompilationContext,
    CompilerStageId,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    execute_shadow,
)
from scorch.stensor import STensor

from tests.test_scorch.test_kernels_comprehensive import assert_close


def dense_stensor(tensor, name):
    return STensor.from_torch(tensor, name).to_dense()


def build_matmul_ikj(dtype=torch.float32):
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    a = TensorVar("A", fmt="dd", dtype=dtype)
    b = TensorVar("B", fmt="dd", dtype=dtype)
    c = TensorVar("C", fmt="dd", dtype=dtype)
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=Operation.ADD
    )
    return ForAll(i, ForAll(k, ForAll(j, assign)))


def build_elementwise(op, dtype=torch.float32):
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd", dtype=dtype)
    b = TensorVar("B", fmt="dd", dtype=dtype)
    c = TensorVar("C", fmt="dd", dtype=dtype)
    assign = TensorAssign(c[i, j], CINBinaryOp(op, a[i, j], b[i, j]))
    return ForAll(i, ForAll(j, assign))


def build_matvec():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    x = TensorVar("x", fmt="d")
    y = TensorVar("y", fmt="d")
    assign = TensorAssign(
        y[i], CINBinaryOp(Operation.MUL, a[i, j], x[j]), op=Operation.ADD
    )
    return ForAll(i, ForAll(j, assign))


def oracle_reference(kernel, args, result_shape):
    """Execute the compiled program's LoopIR through the production oracle."""

    lowering = kernel.lowering
    inputs = {}
    for symbol, tensor in zip(lowering.rhs_access_symbols, args):
        inputs[symbol] = tensor.tolist()
    outputs = run_program(
        lowering.program,
        inputs,
        {lowering.result_symbol: tuple(result_shape)},
    )
    return torch.tensor(outputs[lowering.result_symbol], dtype=torch.float64)


@torch.no_grad()
def test_matmul_shadow_execution_agrees_everywhere():
    torch.manual_seed(20260722)
    cin = build_matmul_ikj()
    a_t, b_t = torch.rand(5, 7), torch.rand(7, 3)
    loopir_result, legacy_result, comparison = execute_shadow(
        cin,
        (5, 3),
        dense_stensor(a_t, "A"),
        dense_stensor(b_t, "B"),
    )
    assert comparison.identical
    loopir_torch = loopir_result.to_torch()
    legacy_torch = legacy_result.to_torch()
    assert torch.equal(loopir_torch, legacy_torch)
    assert_close(loopir_torch, a_t @ b_t)


@torch.no_grad()
def test_elementwise_shadow_execution_agrees_everywhere():
    torch.manual_seed(1)
    cin = build_elementwise(Operation.ADD)
    a_t, b_t = torch.rand(4, 6), torch.rand(4, 6)
    loopir_result, legacy_result, comparison = execute_shadow(
        cin,
        (4, 6),
        dense_stensor(a_t, "A"),
        dense_stensor(b_t, "B"),
    )
    assert comparison.identical
    assert torch.equal(loopir_result.to_torch(), legacy_result.to_torch())
    assert_close(loopir_result.to_torch(), a_t + b_t)


@torch.no_grad()
def test_sub_executes_correctly_without_a_legacy_comparand():
    torch.manual_seed(2)
    cin = build_elementwise(Operation.SUB)
    a_t, b_t = torch.rand(3, 5), torch.rand(3, 5)
    result, kernel = execute_cin_via_loopir(
        cin, (3, 5), dense_stensor(a_t, "A"), dense_stensor(b_t, "B")
    )
    assert_close(result.to_torch(), a_t - b_t)
    oracle = oracle_reference(kernel, (a_t, b_t), (3, 5))
    assert_close(result.to_torch(), oracle.to(torch.float32))


@torch.no_grad()
def test_matvec_executes_and_matches_oracle():
    torch.manual_seed(3)
    cin = build_matvec()
    a_t, x_t = torch.rand(6, 4), torch.rand(4)
    result, kernel = execute_cin_via_loopir(
        cin, (6,), dense_stensor(a_t, "A"), dense_stensor(x_t, "x")
    )
    assert_close(result.to_torch(), a_t @ x_t)
    oracle = oracle_reference(kernel, (a_t, x_t), (6,))
    assert_close(result.to_torch(), oracle.to(torch.float32))


@torch.no_grad()
def test_matmul_float64_executes_and_matches_oracle():
    torch.manual_seed(4)
    cin = build_matmul_ikj(torch.float64)
    a_t = torch.rand(4, 3, dtype=torch.float64)
    b_t = torch.rand(3, 5, dtype=torch.float64)
    result, kernel = execute_cin_via_loopir(
        cin, (4, 5), dense_stensor(a_t, "A"), dense_stensor(b_t, "B")
    )
    assert_close(result.to_torch(), a_t @ b_t)
    oracle = oracle_reference(kernel, (a_t, b_t), (4, 5))
    assert_close(result.to_torch(), oracle)


def test_loopir_stage_timing_is_recorded():
    options = CompileOptions.from_environment()
    context = CompilationContext(options)
    cin = build_elementwise(Operation.ADD)
    compile_cin_via_loopir(
        cin,
        (3, 4),
        [((3, 4), torch.float32)] * 2,
        compile_options=options,
        compilation_context=context,
    )
    recorded = [record.stage_id for record in context.stage_run_records]
    assert recorded[:3] == [
        CompilerStageId.CIN_TO_LOOPIR_LOWERING,
        CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
        CompilerStageId.LLIR_TO_CPP_GENERATION,
    ]


def test_legacy_stage_sequence_never_contains_loopir_stages():
    from scorch.compiler.loopir.pipeline import legacy_generated_cpp

    options = CompileOptions.from_environment()
    cin = build_elementwise(Operation.ADD)
    # The legacy comparand helper drives the untouched legacy lowering; its
    # context must never record a LoopIR stage.
    legacy_generated_cpp(
        copy.deepcopy(cin),
        (3, 4),
        [((3, 4), torch.float32)] * 2,
        compile_options=options,
    )
    # The legacy default entry allocates its own context internally; the
    # stage identities it begins are locked by test_compiler_stage_timing.
    loopir_stages = {
        CompilerStageId.CIN_TO_LOOPIR_LOWERING,
        CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
    }
    import scorch.ops as ops_module

    source = open(ops_module.__file__).read()
    for stage in loopir_stages:
        assert stage.name not in source


def test_requested_schedule_fails_closed():
    from scorch.exceptions import CompileSpecError
    from scorch.compiler.scheduler import Schedule

    options = CompileOptions.from_environment(
        requested_schedule=Schedule(loop_order=["i", "j"])
    )
    cin = build_elementwise(Operation.ADD)
    with pytest.raises(CompileSpecError):
        compile_cin_via_loopir(
            cin,
            (3, 4),
            [((3, 4), torch.float32)] * 2,
            compile_options=options,
        )
