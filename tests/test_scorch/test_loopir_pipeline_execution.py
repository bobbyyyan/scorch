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


def test_requested_schedules_route_through_the_strangler_entry():
    """Phase 6 moved this boundary deliberately: requested schedules are now
    consumed by the scheduled path instead of rejected wholesale.  Explicit
    supported schedules compile (an identity order reproduces the
    unscheduled source exactly); illegal orders still fail closed at the
    shared scheduler boundary, including through the forced-schedule seam;
    and a legacy-accepted order the migrated families cannot emit (the
    recorded dense-ijk erratum, invalid C++ on the legacy path) fails
    closed with the family's stable code."""

    from scorch.compiler.diagnostics import InvalidSchedule
    from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
    from scorch.compiler.scheduler import Schedule, schedule_force

    options = CompileOptions.from_environment(
        requested_schedule=Schedule(loop_order=["i", "j"])
    )
    cin = build_elementwise(Operation.ADD)
    scheduled_kernel = compile_cin_via_loopir(
        copy.deepcopy(cin),
        (3, 4),
        [((3, 4), torch.float32)] * 2,
        compile_options=options,
    )
    assert scheduled_kernel.schedule is not None
    assert scheduled_kernel.schedule.plan.provenance == "explicit"
    unscheduled_kernel = compile_cin_via_loopir(
        copy.deepcopy(cin),
        (3, 4),
        [((3, 4), torch.float32)] * 2,
    )
    assert unscheduled_kernel.schedule is None
    assert scheduled_kernel.cpp_source == unscheduled_kernel.cpp_source

    with schedule_force(Schedule(loop_order=["j", "i"])):
        with pytest.raises(InvalidSchedule):
            compile_cin_via_loopir(
                copy.deepcopy(cin),
                (3, 4),
                [((3, 4), torch.float32)] * 2,
            )

    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    matmul = ForAll(
        i,
        ForAll(
            k,
            ForAll(
                j,
                TensorAssign(
                    c[i, j],
                    CINBinaryOp(Operation.MUL, a[i, k], b[k, j]),
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    erratum_options = CompileOptions.from_environment(
        requested_schedule=Schedule(loop_order=["i", "j", "k"])
    )
    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(
            matmul,
            (3, 4),
            [((3, 5), torch.float32), ((5, 4), torch.float32)],
            compile_options=erratum_options,
        )
    assert error.value.defect.code == "unsupported_loop_order"


# -- Phase-5 sparse families: compiled execution differentials ----------------

from scorch.compiler.loopir.levels import CsrMatrix  # noqa: E402


def sparse_stensor(tensor, name):
    return STensor.from_torch(tensor, name).to_sparse("ds")


def build_spmv_csr(dtype=torch.float32):
    i, j = IndexVar("i"), IndexVar("j")
    y = TensorVar("y", fmt="d", dtype=dtype)
    a = TensorVar("A", fmt="ds", dtype=dtype)
    x = TensorVar("x", fmt="d", dtype=dtype)
    assign = TensorAssign(
        y[i], CINBinaryOp(Operation.MUL, a[i, j], x[j]), op=Operation.ADD
    )
    return ForAll(i, ForAll(j, assign))


def build_spmm_csr_dense(update_op=Operation.ADD, dtype=torch.float32):
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    c = TensorVar("C", fmt="dd", dtype=dtype)
    a = TensorVar("A", fmt="ds", dtype=dtype)
    b = TensorVar("B", fmt="dd", dtype=dtype)
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=update_op
    )
    return ForAll(i, ForAll(k, ForAll(j, assign)))


def build_sparse_elementwise_cin(op, fmt_out):
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt=fmt_out)
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    assign = TensorAssign(c[i, j], CINBinaryOp(op, a[i, j], b[i, j]))
    return ForAll(i, ForAll(j, assign))


def masked_random(rows, cols, density, *, empty_rows=(), dtype=torch.float32):
    dense = torch.rand(rows, cols, dtype=dtype)
    dense = dense * (torch.rand(rows, cols) < density)
    for row in empty_rows:
        dense[row] = 0.0
    return dense


def sparse_oracle_reference(kernel, dense_args, arg_formats, result_shape):
    """Execute the compiled program's LoopIR through the production oracle."""

    lowering = kernel.lowering
    inputs = {}
    for symbol, tensor, fmt in zip(
        lowering.rhs_access_symbols, dense_args, arg_formats
    ):
        listed = tensor.to(torch.float64).tolist()
        if fmt == "ds":
            inputs[symbol] = CsrMatrix.from_dense(listed)
        else:
            inputs[symbol] = listed
    return run_program(
        lowering.program,
        inputs,
        {lowering.result_symbol: tuple(result_shape)},
    )


def csr_result_parts(result):
    """Snapshot a CSR STensor's storage before any densifying access."""

    positions, coordinates = result.index.mode_indices[1]
    return (
        [int(v) for v in positions.tolist()],
        [int(v) for v in coordinates.tolist()],
        result.values.clone(),
    )


@torch.no_grad()
def test_spmv_csr_executes_and_matches_oracle_and_torch():
    torch.manual_seed(31)
    cin = build_spmv_csr()
    a_t = masked_random(6, 9, 0.4, empty_rows=(2, 5))
    x_t = torch.rand(9)
    result, kernel = execute_cin_via_loopir(
        copy.deepcopy(cin), (6,), sparse_stensor(a_t, "A"), dense_stensor(x_t, "x")
    )
    assert_close(result.to_torch(), a_t @ x_t)
    oracle = sparse_oracle_reference(kernel, (a_t, x_t), ("ds", "d"), (6,))
    assert_close(
        result.to_torch(),
        torch.tensor(oracle[kernel.lowering.result_symbol]).to(torch.float32),
    )


@torch.no_grad()
def test_spmv_csr_float64_executes():
    torch.manual_seed(32)
    cin = build_spmv_csr(dtype=torch.float64)
    a_t = masked_random(5, 4, 0.5, dtype=torch.float64)
    x_t = torch.rand(4, dtype=torch.float64)
    result, _ = execute_cin_via_loopir(
        copy.deepcopy(cin), (5,), sparse_stensor(a_t, "A"), dense_stensor(x_t, "x")
    )
    assert_close(result.to_torch(), a_t @ x_t)


@torch.no_grad()
def test_spmm_csr_dense_shadow_execution_agrees_everywhere():
    torch.manual_seed(33)
    cin = build_spmm_csr_dense()
    a_t = masked_random(6, 9, 0.35, empty_rows=(0,))
    b_t = torch.rand(9, 4)
    loopir_result, legacy_result, comparison = execute_shadow(
        cin,
        (6, 4),
        sparse_stensor(a_t, "A"),
        dense_stensor(b_t, "B"),
    )
    assert comparison.identical
    assert torch.equal(loopir_result.to_torch(), legacy_result.to_torch())
    assert_close(loopir_result.to_torch(), a_t @ b_t)
    kernel = compile_cin_via_loopir(
        copy.deepcopy(cin),
        (6, 4),
        [((6, 9), torch.float32), ((9, 4), torch.float32)],
    )
    oracle = sparse_oracle_reference(kernel, (a_t, b_t), ("ds", "dd"), (6, 4))
    assert_close(
        loopir_result.to_torch(),
        torch.tensor(oracle[kernel.lowering.result_symbol]).to(torch.float32),
    )


@torch.no_grad()
def test_public_implicit_spmm_bridges_and_agrees_everywhere():
    """The frontend's op=None SpMM matches its explicit-ADD twin end to end.

    The public einsum/matmul frontend leaves ``TensorAssign.op`` unset for
    reductions; the CIN-to-LoopIR ownership boundary normalizes the proven
    additive update once.  The bridged program must be byte-identical to
    the explicit spelling on both routes, execute identically, and match
    the legacy shadow and the PyTorch reference.
    """

    torch.manual_seed(37)
    implicit = build_spmm_csr_dense(update_op=None)
    explicit = build_spmm_csr_dense()
    bindings = [((6, 9), torch.float32), ((9, 4), torch.float32)]
    implicit_kernel = compile_cin_via_loopir(copy.deepcopy(implicit), (6, 4), bindings)
    explicit_kernel = compile_cin_via_loopir(copy.deepcopy(explicit), (6, 4), bindings)
    assert implicit_kernel.cpp_source == explicit_kernel.cpp_source

    a_t = masked_random(6, 9, 0.35, empty_rows=(0, 4))
    b_t = torch.rand(9, 4)
    loopir_result, legacy_result, comparison = execute_shadow(
        implicit,
        (6, 4),
        sparse_stensor(a_t, "A"),
        dense_stensor(b_t, "B"),
    )
    assert comparison.identical
    assert torch.equal(loopir_result.to_torch(), legacy_result.to_torch())
    assert_close(loopir_result.to_torch(), a_t @ b_t)


@torch.no_grad()
def test_public_implicit_spmm_bridges_for_float64_and_empty_extents():
    implicit64 = build_spmm_csr_dense(update_op=None, dtype=torch.float64)
    explicit64 = build_spmm_csr_dense(dtype=torch.float64)
    bindings64 = [((5, 7), torch.float64), ((7, 3), torch.float64)]
    implicit_kernel = compile_cin_via_loopir(
        copy.deepcopy(implicit64), (5, 3), bindings64
    )
    explicit_kernel = compile_cin_via_loopir(
        copy.deepcopy(explicit64), (5, 3), bindings64
    )
    assert implicit_kernel.cpp_source == explicit_kernel.cpp_source

    kernel = compile_cin_via_loopir(
        build_spmm_csr_dense(update_op=None),
        (0, 4),
        [((0, 9), torch.float32), ((9, 4), torch.float32)],
    )
    lowering = kernel.lowering
    sparse_symbol, dense_symbol = lowering.rhs_access_symbols
    dense = [[float(row + column) for column in range(4)] for row in range(9)]
    result = run_program(
        lowering.program,
        {
            sparse_symbol: CsrMatrix(0, 9, (0,), (), ()),
            dense_symbol: dense,
        },
        {lowering.result_symbol: (0, 4)},
    )
    assert result[lowering.result_symbol] == []


def test_spmm_oracle_preserves_hidden_extent_with_zero_rows():
    cin = build_spmm_csr_dense()
    kernel = compile_cin_via_loopir(
        cin,
        (0, 4),
        [((0, 9), torch.float32), ((9, 4), torch.float32)],
    )
    lowering = kernel.lowering
    sparse_symbol, dense_symbol = lowering.rhs_access_symbols
    dense = [[float(row + column) for column in range(4)] for row in range(9)]
    result = run_program(
        lowering.program,
        {
            sparse_symbol: CsrMatrix(0, 9, (0,), (), ()),
            dense_symbol: dense,
        },
        {lowering.result_symbol: (0, 4)},
    )

    assert result[lowering.result_symbol] == []


@torch.no_grad()
@pytest.mark.parametrize(
    "operation,reference",
    [(Operation.ADD, torch.add), (Operation.MUL, torch.mul)],
    ids=["union_add", "intersection_mul"],
)
def test_sparse_elementwise_to_dense_shadow_agrees_everywhere(operation, reference):
    torch.manual_seed(34)
    cin = build_sparse_elementwise_cin(operation, "dd")
    a_t = masked_random(5, 8, 0.4, empty_rows=(1,))
    b_t = masked_random(5, 8, 0.4, empty_rows=(1, 3))
    loopir_result, legacy_result, comparison = execute_shadow(
        cin,
        (5, 8),
        sparse_stensor(a_t, "A"),
        sparse_stensor(b_t, "B"),
    )
    assert comparison.identical
    assert torch.equal(loopir_result.to_torch(), legacy_result.to_torch())
    assert_close(loopir_result.to_torch(), reference(a_t, b_t))


@torch.no_grad()
@pytest.mark.parametrize(
    "operation,reference",
    [(Operation.ADD, torch.add), (Operation.MUL, torch.mul)],
    ids=["union_add", "intersection_mul"],
)
def test_sparse_elementwise_to_csr_matches_oracle_structure(operation, reference):
    """Compiled CSR assembly must equal the oracle's ordered append stream."""

    torch.manual_seed(35)
    cin = build_sparse_elementwise_cin(operation, "ds")
    # Disjoint columns in row 0, overlap in row 2, one-sided exhaustion in
    # rows 3 (A only) and 4 (B only), and a fully empty row 1.
    a_t = torch.zeros(5, 6)
    b_t = torch.zeros(5, 6)
    a_t[0, 0], b_t[0, 5] = 1.5, 2.5
    a_t[2, 1], b_t[2, 1] = 3.0, 4.0
    a_t[2, 4], b_t[2, 2] = -1.0, 0.5
    a_t[3, 3] = 7.0
    b_t[4, 0], b_t[4, 5] = -2.0, 6.0
    result, kernel = execute_cin_via_loopir(
        copy.deepcopy(cin), (5, 6), sparse_stensor(a_t, "A"), sparse_stensor(b_t, "B")
    )
    positions, coordinates, values = csr_result_parts(result)
    oracle = sparse_oracle_reference(kernel, (a_t, b_t), ("ds", "ds"), (5, 6))
    produced = oracle[kernel.lowering.result_symbol]
    assert positions == list(produced.indptr)
    assert coordinates == list(produced.indices)
    assert_close(values, torch.tensor(produced.values).to(torch.float32))
    assert_close(result.to_torch(), reference(a_t, b_t))


def test_shadow_execution_rejects_sparse_outputs_at_its_public_boundary():
    cin = build_sparse_elementwise_cin(Operation.ADD, "ds")
    a = sparse_stensor(torch.tensor([[1.0, 0.0]]), "A")
    b = sparse_stensor(torch.tensor([[0.0, 2.0]]), "B")

    with pytest.raises(CompileSpecError, match="requires dense outputs"):
        execute_shadow(cin, (1, 2), a, b)


@torch.no_grad()
def test_union_add_to_csr_keeps_exact_zero_cancellation():
    cin = build_sparse_elementwise_cin(Operation.ADD, "ds")
    a_t = torch.tensor([[2.5, 0.0, 1.0]])
    b_t = torch.tensor([[-2.5, 0.0, 4.0]])
    result, _ = execute_cin_via_loopir(
        copy.deepcopy(cin), (1, 3), sparse_stensor(a_t, "A"), sparse_stensor(b_t, "B")
    )
    positions, coordinates, values = csr_result_parts(result)
    assert positions == [0, 2]
    assert coordinates == [0, 2]
    assert values.tolist() == [0.0, 5.0]


@torch.no_grad()
def test_sparse_families_execute_on_empty_and_zero_extent_inputs():
    add_cin = build_sparse_elementwise_cin(Operation.ADD, "ds")
    empty_a = torch.zeros(4, 3)
    empty_b = torch.zeros(4, 3)
    result, _ = execute_cin_via_loopir(
        copy.deepcopy(add_cin),
        (4, 3),
        sparse_stensor(empty_a, "A"),
        sparse_stensor(empty_b, "B"),
    )
    positions, coordinates, values = csr_result_parts(result)
    assert positions == [0, 0, 0, 0, 0]
    assert coordinates == []
    assert values.numel() == 0

    spmv_cin = build_spmv_csr()
    zero_rows = torch.zeros(0, 4)
    result, _ = execute_cin_via_loopir(
        copy.deepcopy(spmv_cin),
        (0,),
        sparse_stensor(zero_rows, "A"),
        dense_stensor(torch.rand(4), "x"),
    )
    assert result.to_torch().numel() == 0


@torch.no_grad()
def test_sparse_families_match_torch_on_random_grids():
    torch.manual_seed(36)
    add_cin = build_sparse_elementwise_cin(Operation.ADD, "ds")
    mul_cin = build_sparse_elementwise_cin(Operation.MUL, "ds")
    for rows, cols, density in [(3, 5, 0.3), (7, 4, 0.5), (6, 8, 0.15)]:
        a_t = masked_random(rows, cols, density)
        b_t = masked_random(rows, cols, density)
        added, _ = execute_cin_via_loopir(
            copy.deepcopy(add_cin),
            (rows, cols),
            sparse_stensor(a_t, "A"),
            sparse_stensor(b_t, "B"),
        )
        assert_close(added.to_torch(), a_t + b_t)
        multiplied, _ = execute_cin_via_loopir(
            copy.deepcopy(mul_cin),
            (rows, cols),
            sparse_stensor(a_t, "A"),
            sparse_stensor(b_t, "B"),
        )
        assert_close(multiplied.to_torch(), a_t * b_t)


@torch.no_grad()
def test_sparse_jit_rejects_forged_duplicate_coordinates():
    cin = build_spmv_csr()
    a = sparse_stensor(torch.tensor([[1.0, 2.0]]), "A")
    # Public storage construction rejects duplicates.  Forge the owned native
    # array afterward to lock the deeper JIT boundary against stale/corrupt
    # runtime state.
    a.storage._mode_indices[1][1][1] = 0

    with pytest.raises(RuntimeError, match="strictly increasing"):
        execute_cin_via_loopir(
            cin,
            (1,),
            a,
            dense_stensor(torch.ones(2), "x"),
        )


@torch.no_grad()
def test_sparse_stage_timing_is_recorded():
    cin = build_spmv_csr()
    options = CompileOptions.from_environment()
    context = CompilationContext(options)
    a_t = masked_random(4, 5, 0.5)
    x_t = torch.rand(5)
    execute_cin_via_loopir(
        copy.deepcopy(cin),
        (4,),
        sparse_stensor(a_t, "A"),
        dense_stensor(x_t, "x"),
        compile_options=options,
        _compilation_context=context,
    )
    stage_ids = [record.stage_id for record in context.stage_run_records]
    assert CompilerStageId.CIN_TO_LOOPIR_LOWERING in stage_ids
    assert CompilerStageId.LOOPIR_TO_LLIR_LOWERING in stage_ids
    assert stage_ids.index(CompilerStageId.CIN_TO_LOOPIR_LOWERING) < stage_ids.index(
        CompilerStageId.LOOPIR_TO_LLIR_LOWERING
    )


def test_panel_scheduled_stage_timing_is_recorded():
    """A panel schedule owns the full scheduled stage sequence, including
    the LoopIR schedule-application stage, and the managed pass records."""

    from scorch.compiler.scheduler import Schedule, TileSpec

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    spmm = ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(
                    c[i, k],
                    CINBinaryOp(Operation.MUL, a[i, j], b[j, k]),
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(TileSpec("j", 3, kind="panel", accum="direct"),),
        tag="panel-stages",
        parallel_loop="i",
    )
    options = CompileOptions.from_environment(requested_schedule=schedule)
    context = CompilationContext(options)
    kernel = compile_cin_via_loopir(
        spmm,
        (4, 6),
        (((4, 5), torch.float32), ((5, 6), torch.float32)),
        compile_options=options,
        compilation_context=context,
    )
    assert kernel.schedule is not None
    recorded = [record.stage_id for record in context.stage_run_records]
    assert recorded == [
        CompilerStageId.CIN_NORMALIZATION_AND_VERIFICATION,
        CompilerStageId.SCHEDULING_AND_LOOP_PLAN_CONSTRUCTION,
        CompilerStageId.FRONTEND_VALIDATED_OPERATION_CONSTRUCTION,
        CompilerStageId.CIN_TO_LOOPIR_LOWERING,
        CompilerStageId.LOOPIR_SCHEDULE_APPLICATION,
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
