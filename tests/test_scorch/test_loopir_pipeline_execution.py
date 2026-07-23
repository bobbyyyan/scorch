"""End-to-end LoopIR dense vertical slice: compile, execute, and compare.

Curated differential execution for the migrated families: the LoopIR path's
kernel must agree bitwise with the untouched legacy path (their generated
sources are byte-identical, so they share one kernel artifact), match the
PyTorch reference within the repository tolerance convention, and match the
production LoopIR oracle.  This module is the only place that runs both
pipelines on real tensors; nothing enables shadow execution globally.
"""

import copy
from dataclasses import replace

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
    CompilationContextError,
    CompilerStageId,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.exceptions import CompileSpecError
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
def test_runtime_binding_relayouts_nonidentity_dense_inputs():
    cin = build_elementwise(Operation.ADD)
    a_t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b_t = torch.arange(6, 12, dtype=torch.float32).reshape(2, 3)
    a = STensor.from_torch(a_t, "A", mode_order=[1, 0]).to_dense()
    b = STensor.from_torch(b_t, "B", mode_order=[1, 0]).to_dense()
    result, _ = execute_cin_via_loopir(cin, (2, 3), a, b)
    assert torch.equal(result.to_torch(), a_t + b_t)


def test_runtime_binding_rejects_format_mismatch_before_native_execution():
    cin = build_elementwise(Operation.ADD)
    sparse = STensor.from_torch(torch.eye(2).to_sparse_csr(), "A")
    dense = dense_stensor(torch.ones(2, 2), "B")
    with pytest.raises(CompileSpecError, match="expects format"):
        execute_cin_via_loopir(cin, (2, 2), sparse, dense)


def test_format_mismatch_is_rejected_before_mode_order_relayout(monkeypatch):
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="ddd")
    b = TensorVar("B", fmt="ddd")
    c = TensorVar("C", fmt="ddd")
    cin = ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(
                    c[i, j, k], CINBinaryOp(Operation.ADD, a[i, j, k], b[i, j, k])
                ),
            ),
        ),
    )
    source = torch.ones(2, 3, 4)
    sparse = STensor.from_torch(
        source.to_sparse_coo(),
        "A",
        mode_order=[1, 0, 2],
    )

    def unexpected_relayout(*args, **kwargs):
        raise AssertionError("format mismatch reached mode-order relayout")

    monkeypatch.setattr(STensor, "change_mode_order", unexpected_relayout)
    with pytest.raises(CompileSpecError, match="expects format"):
        execute_cin_via_loopir(
            cin,
            (2, 3, 4),
            sparse,
            dense_stensor(source, "B"),
        )
    with pytest.raises(CompileSpecError, match="expects format"):
        execute_shadow(
            cin,
            (2, 3, 4),
            sparse,
            dense_stensor(source, "B"),
        )


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
        compilation_context=context,
    )
    recorded = [record.stage_id for record in context.stage_run_records]
    assert recorded == [
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION,
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        CompilerStageId.CIN_TO_LOOPIR_LOWERING,
        CompilerStageId.LOOPIR_TO_LLIR_LOWERING,
        CompilerStageId.LLIR_TO_CPP_GENERATION,
    ]
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]


def test_partial_target_failure_preserves_pass_prefix_and_retires_context(
    monkeypatch,
):
    from scorch.compiler.llir_pass_manager import LLIRPassManager

    options = CompileOptions.from_environment()
    context = CompilationContext(options)

    def fail_dense_pointer_pass(*args, **kwargs):
        raise RuntimeError("injected dense-pointer failure")

    monkeypatch.setattr(
        LLIRPassManager,
        "run_dense_pointer_hoist",
        fail_dense_pointer_pass,
    )
    with pytest.raises(RuntimeError, match="injected dense-pointer failure"):
        compile_cin_via_loopir(
            build_elementwise(Operation.ADD),
            (3, 4),
            [((3, 4), torch.float32)] * 2,
            compile_options=options,
            compilation_context=context,
        )
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch"
    ]
    assert CompilerStageId.LOOPIR_TO_LLIR_LOWERING not in {
        record.stage_id for record in context.stage_run_records
    }
    with pytest.raises(CompilationContextError) as error:
        context.begin_stage(
            CompilerStageId.LLIR_TO_CPP_GENERATION,
            compile_options=options,
        )
    assert error.value.diagnostic.code == "failed_compilation"


def test_source_comparison_resolves_environment_options_once(monkeypatch):
    from scorch.compiler.loopir.pipeline import compare_generated_sources

    options = CompileOptions.from_environment()
    calls = 0

    def from_environment(cls, *args, **kwargs):
        nonlocal calls
        calls += 1
        return options

    monkeypatch.setattr(
        CompileOptions,
        "from_environment",
        classmethod(from_environment),
    )
    comparison = compare_generated_sources(
        build_elementwise(Operation.ADD),
        (2, 3),
        [((2, 3), torch.float32)] * 2,
    )
    assert comparison.identical
    assert calls == 1


def test_shadow_routes_one_exact_options_snapshot(monkeypatch):
    import scorch.ops as ops_module

    options = replace(CompileOptions.from_environment(), emit_comments=False)
    original = ops_module.lower_and_exec_cin
    captured = {}

    def wrapped(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(ops_module, "lower_and_exec_cin", wrapped)
    cin = build_elementwise(Operation.ADD)
    tensor = dense_stensor(torch.ones(2, 2), "A")
    execute_shadow(
        cin,
        (2, 2),
        tensor,
        dense_stensor(torch.ones(2, 2), "B"),
        compile_options=options,
    )
    assert captured["_compile_options"] is options


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
    from scorch.compiler.scheduler import Schedule, schedule_force

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
    with schedule_force(Schedule(loop_order=["j", "i"])):
        with pytest.raises(CompileSpecError):
            compile_cin_via_loopir(
                cin,
                (3, 4),
                [((3, 4), torch.float32)] * 2,
            )
