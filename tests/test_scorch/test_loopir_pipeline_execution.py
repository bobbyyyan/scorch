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
    IndexStmt,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    UnaryOp,
    Where,
    Workspace,
)
from scorch.compiler.compilation_context import (
    CompilationContext,
    CompilationContextError,
    CompilerStageId,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import VerificationError
from scorch.exceptions import CompileSpecError
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compare_generated_sources,
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    execute_shadow,
    legacy_generated_cpp,
)
from scorch.format import TensorFormat
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
    result_decl = next(
        decl
        for decl in lowering.program.tensors
        if decl.symbol == lowering.result_symbol
    )
    logical_shape = [0] * len(result_decl.levels)
    for physical_level, level in enumerate(result_decl.levels):
        logical_shape[level.mode] = result_shape[physical_level]
    outputs = run_program(
        lowering.program,
        inputs,
        {lowering.result_symbol: tuple(logical_shape)},
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
    result, _ = execute_cin_via_loopir(cin, torch.Size((2, 3)), a, b)
    assert torch.equal(result.to_torch(), a_t + b_t)


def test_runtime_binding_rejects_format_mismatch_before_native_execution():
    cin = build_elementwise(Operation.ADD)
    sparse = STensor.from_torch(torch.eye(2).to_sparse_csr(), "A")
    dense = dense_stensor(torch.ones(2, 2), "B")
    with pytest.raises(CompileSpecError, match="expects format"):
        execute_cin_via_loopir(cin, (2, 2), sparse, dense)


@pytest.mark.parametrize("entry_name", ("loopir", "legacy"))
def test_runtime_entries_reject_non_stensor_args_without_descriptor_access(
    entry_name,
):
    """A hostile later operand is rejected before any operand property read."""

    from scorch.ops import lower_and_exec_cin

    reads = 0

    class HostileOperand:
        def __getattribute__(self, name):
            nonlocal reads
            if name != "__class__":
                reads += 1
            raise RuntimeError(f"hostile runtime descriptor {name}")

    args = (
        dense_stensor(torch.ones(2, 2), "A"),
        HostileOperand(),
    )
    entry = execute_cin_via_loopir if entry_name == "loopir" else lower_and_exec_cin

    with pytest.raises(CompileSpecError, match="exact STensor"):
        entry(build_elementwise(Operation.ADD), (2, 2), *args)

    assert reads == 0


@pytest.mark.parametrize("entry_name", ("loopir", "legacy"))
def test_runtime_entries_wrap_forged_exact_stensor_state(entry_name):
    """Exact-class forgeries fail as public compile specifications."""

    from scorch.ops import lower_and_exec_cin

    forged = object.__new__(STensor)
    args = (
        dense_stensor(torch.ones(2, 2), "A"),
        forged,
    )
    entry = execute_cin_via_loopir if entry_name == "loopir" else lower_and_exec_cin

    with pytest.raises(CompileSpecError, match="invalid internal state"):
        entry(build_elementwise(Operation.ADD), (2, 2), *args)


@pytest.mark.parametrize("entry_name", ("loopir", "legacy"))
def test_runtime_surplus_hostile_operand_is_not_inspected(entry_name):
    """Normalized CIN arity wins before any surplus operand inspection."""

    from scorch.ops import lower_and_exec_cin

    reads = 0

    class HostileSurplus:
        def __getattribute__(self, name):
            nonlocal reads
            if name != "__class__":
                reads += 1
            raise RuntimeError(f"surplus operand was inspected through {name}")

    args = (
        dense_stensor(torch.ones(2, 2), "A"),
        dense_stensor(torch.ones(2, 2), "B"),
        HostileSurplus(),
    )
    entry = execute_cin_via_loopir if entry_name == "loopir" else lower_and_exec_cin

    with pytest.raises(CompileSpecError, match="expects 2 runtime tensors, got 3"):
        entry(build_elementwise(Operation.ADD), (2, 2), *args)

    assert reads == 0


@pytest.mark.parametrize("entry", (compile_cin_via_loopir, legacy_generated_cpp))
def test_source_surplus_hostile_binding_is_not_inspected(entry):
    """Compile-only binding arity also precedes binding-item validation."""

    reads = 0

    class HostileBinding:
        def __getattribute__(self, name):
            nonlocal reads
            if name != "__class__":
                reads += 1
            raise RuntimeError(f"surplus binding was inspected through {name}")

    bindings = [
        ((2, 2), torch.float32),
        ((2, 2), torch.float32),
        HostileBinding(),
    ]

    with pytest.raises(CompileSpecError, match="expects 2 runtime tensors, got 3"):
        entry(build_elementwise(Operation.ADD), (2, 2), bindings)

    assert reads == 0


def test_loopir_runtime_binding_failure_makes_explicit_context_terminal():
    """Runtime shape/arity validation is owned by the frontend binding stage."""

    options = CompileOptions.from_environment()
    context = CompilationContext(options)

    with pytest.raises(CompileSpecError, match="expects 2 runtime tensors, got 1"):
        execute_cin_via_loopir(
            build_elementwise(Operation.ADD),
            (2, 2),
            dense_stensor(torch.ones(2, 2), "A"),
            compile_options=options,
            _compilation_context=context,
        )

    with pytest.raises(CompilationContextError) as terminal:
        context.begin_stage(
            CompilerStageId.LEGACY_CIN_ADAPTATION,
            compile_options=options,
        )
    assert terminal.value.diagnostic.code == "failed_compilation"


@pytest.mark.parametrize("entry_name", ("loopir", "legacy"))
def test_post_relayout_snapshot_failure_makes_context_terminal(
    monkeypatch,
    entry_name,
):
    """Aligned operands validate before CIN mutation under a failure owner."""

    import scorch.ops as ops_module

    options = CompileOptions.from_environment()
    context = CompilationContext(options)
    forged = object.__new__(STensor)

    def forge_aligned(runtime_args, *args, **kwargs):
        return (forged, *runtime_args[1:])

    monkeypatch.setattr(
        ops_module,
        "_relayout_mode_order_args",
        forge_aligned,
    )
    cin = build_elementwise(Operation.ADD)
    runtime_args = (
        dense_stensor(torch.ones(2, 2), "A"),
        dense_stensor(torch.ones(2, 2), "B"),
    )

    with pytest.raises(CompileSpecError, match="invalid internal state"):
        if entry_name == "loopir":
            execute_cin_via_loopir(
                cin,
                (2, 2),
                *runtime_args,
                compile_options=options,
                _compilation_context=context,
            )
        else:
            ops_module.lower_and_exec_cin(
                cin,
                (2, 2),
                *runtime_args,
                _compile_options=options,
                _compilation_context=context,
            )

    with pytest.raises(CompilationContextError) as terminal:
        context.begin_stage(
            CompilerStageId.LEGACY_CIN_ADAPTATION,
            compile_options=options,
        )
    assert terminal.value.diagnostic.code == "failed_compilation"


def test_direct_relayout_failure_makes_context_terminal(monkeypatch):
    """A prerequisite failure between direct binding stages retires its context."""

    import scorch.ops as ops_module

    options = CompileOptions.from_environment()
    context = CompilationContext(options)
    error = RuntimeError("injected direct relayout failure")

    def fail_relayout(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        ops_module,
        "_relayout_mode_order_args",
        fail_relayout,
    )

    with pytest.raises(RuntimeError) as failure:
        ops_module.lower_and_exec_cin(
            build_elementwise(Operation.ADD),
            (2, 2),
            dense_stensor(torch.ones(2, 2), "A"),
            dense_stensor(torch.ones(2, 2), "B"),
            _compile_options=options,
            _compilation_context=context,
        )

    assert failure.value is error
    with pytest.raises(CompilationContextError) as terminal:
        context.begin_stage(
            CompilerStageId.LEGACY_CIN_ADAPTATION,
            compile_options=options,
        )
    assert terminal.value.diagnostic.code == "failed_compilation"


def test_runtime_rejects_forged_dense_storage_cardinality():
    """Dense physical shape and value cardinality agree before pybind."""

    forged = dense_stensor(torch.ones(2, 2), "A")
    object.__setattr__(
        forged.storage,
        "_value",
        torch.ones(3, dtype=torch.float32),
    )

    with pytest.raises(CompileSpecError, match="storage cardinality"):
        execute_cin_via_loopir(
            build_elementwise(Operation.ADD),
            (2, 2),
            forged,
            dense_stensor(torch.ones(2, 2), "B"),
        )


def test_direct_result_rank_failure_precedes_metadata_and_lowering(monkeypatch):
    """Direct execution validates its one-result format contract in-stage."""

    import scorch.ops as ops_module

    options = CompileOptions.from_environment()
    context = CompilationContext(options)
    lowering_calls = 0

    def unexpected_lowering(*args, **kwargs):
        nonlocal lowering_calls
        lowering_calls += 1
        raise AssertionError("invalid result rank reached legacy lowering")

    monkeypatch.setattr(
        ops_module.CINLowerer,
        "_lower_owned_IndexStmt",
        unexpected_lowering,
    )
    cin = build_elementwise(Operation.ADD)
    a = dense_stensor(torch.ones(2, 2), "A")
    b = dense_stensor(torch.ones(2, 2), "B")

    with pytest.raises(CompileSpecError, match="result format rank"):
        ops_module.lower_and_exec_cin(
            cin,
            (4,),
            a,
            b,
            _compile_options=options,
            _compilation_context=context,
        )

    assert lowering_calls == 0
    with pytest.raises(CompilationContextError) as terminal:
        context.begin_stage(
            CompilerStageId.LEGACY_CIN_ADAPTATION,
            compile_options=options,
        )
    assert terminal.value.diagnostic.code == "failed_compilation"


def test_direct_result_contract_ignores_workspace_results(monkeypatch):
    """One external result remains valid when a Where also writes a workspace."""

    import scorch.ops as ops_module

    class AcceptedWorkspaceResult(Exception):
        pass

    def stop_after_result_contract(*args, **kwargs):
        raise AcceptedWorkspaceResult

    monkeypatch.setattr(
        ops_module,
        "_relayout_mode_order_args",
        stop_after_result_contract,
    )
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    workspace = Workspace("wksp", dim=1, dense=True)
    cin = ForAll(
        i,
        Where(
            producer=ForAll(
                k,
                ForAll(
                    j,
                    TensorAssign(
                        workspace[j],
                        CINBinaryOp(Operation.MUL, a[i, k], b[k, j]),
                        op=Operation.ADD,
                    ),
                ),
            ),
            consumer=ForAll(j, TensorAssign(c[i, j], workspace[j])),
        ),
    )

    with pytest.raises(AcceptedWorkspaceResult):
        ops_module.lower_and_exec_cin(
            cin,
            (2, 4),
            dense_stensor(torch.ones(2, 3), "A"),
            dense_stensor(torch.ones(3, 4), "B"),
        )


def test_direct_runtime_late_format_mismatch_precedes_relayout_and_cin_mutation(
    monkeypatch,
):
    """All operands validate before relayout or compiler-owned metadata writes."""

    import scorch.ops as ops_module

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    cin = ForAll(
        j,
        ForAll(
            i,
            TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j])),
        ),
    )
    before = (tuple(a.mode_order), tuple(b.mode_order), tuple(c.mode_order))
    monkeypatch.setattr(ops_module, "normalize_cin", lambda program, **kwargs: program)
    relayout_calls = 0

    def unexpected_relayout(*args, **kwargs):
        nonlocal relayout_calls
        relayout_calls += 1
        raise AssertionError("format mismatch reached runtime relayout")

    monkeypatch.setattr(
        ops_module,
        "_relayout_mode_order_args",
        unexpected_relayout,
    )
    valid = dense_stensor(torch.ones(2, 3), "A")
    late_mismatch = STensor.from_torch(
        torch.ones(2, 3).to_sparse_csr(),
        "B",
    )

    with pytest.raises(CompileSpecError, match="expects format"):
        ops_module.lower_and_exec_cin(cin, (2, 3), valid, late_mismatch)

    assert relayout_calls == 0
    assert (tuple(a.mode_order), tuple(b.mode_order), tuple(c.mode_order)) == before


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


def _reduce_out_options(regblock_enabled):
    from scorch.compiler.scheduler import Schedule

    return replace(
        CompileOptions.from_environment().with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(),
    )


@torch.no_grad()
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_reduce_out_matmul_matches_legacy_bytes_and_execution(regblock_enabled):
    """The automatic dense reduce-out family agrees with legacy end to end.

    The empty-Schedule route carries the strip-mined reduction producer
    plus accumulate copy-out consumer through LoopIR; the generated source
    must be byte-identical to the legacy automatic surgery (including the
    tile-count parallel work estimate on the off arm), and the compiled
    kernel must execute bitwise-identically to the legacy production auto
    route and match the PyTorch reference.
    """

    from scorch.ops import lower_and_exec_cin

    torch.manual_seed(26)
    options = _reduce_out_options(regblock_enabled)
    cin = build_matmul_ikj()
    comparison = compare_generated_sources(
        copy.deepcopy(cin),
        (4, 6),
        (((4, 5), torch.float32), ((5, 6), torch.float32)),
        compile_options=options,
    )
    assert comparison.identical
    width = 8 if regblock_enabled else 32
    assert f"constexpr int kTile_j = {width};" in comparison.loopir_cpp
    assert f"constexpr int kTile_k = {width};" in comparison.loopir_cpp
    assert "wksp[j_in] +=" in comparison.loopir_cpp
    assert "+= wksp[j_in];" in comparison.loopir_cpp
    if not regblock_enabled:
        assert "(B1_size + kTile_j - 1) / kTile_j" in comparison.loopir_cpp

    a_t, b_t = torch.rand(4, 5), torch.rand(5, 6)
    loopir_result, _ = execute_cin_via_loopir(
        copy.deepcopy(cin),
        (4, 6),
        dense_stensor(a_t, "A"),
        dense_stensor(b_t, "B"),
        compile_options=options,
    )
    legacy_result = lower_and_exec_cin(
        copy.deepcopy(cin),
        (4, 6),
        dense_stensor(a_t, "A"),
        dense_stensor(b_t, "B"),
        _compile_options=CompileOptions.from_environment().with_regblock_enabled(
            regblock_enabled
        ),
    )
    assert torch.equal(loopir_result.to_torch(), legacy_result.to_torch())
    assert_close(loopir_result.to_torch(), a_t @ b_t)


@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_reduce_out_family_byte_parity_across_shapes(regblock_enabled):
    """Ragged, oversized, unit, zero, f64, implicit, and rank-3 cells."""

    def build_ttm():
        i, j, ell, k = (IndexVar(n) for n in ("i", "j", "ell", "k"))
        a = TensorVar("A", fmt="ddd")
        b = TensorVar("B", fmt="dd")
        c = TensorVar("C", fmt="ddd")
        assign = TensorAssign(
            c[i, j, k],
            CINBinaryOp(Operation.MUL, a[i, j, ell], b[ell, k]),
            op=Operation.ADD,
        )
        return ForAll(i, ForAll(j, ForAll(ell, ForAll(k, assign))))

    options = _reduce_out_options(regblock_enabled)
    cells = [
        (
            build_matmul_ikj(),
            (3, 13),
            (((3, 9), torch.float32), ((9, 13), torch.float32)),
        ),
        (
            build_matmul_ikj(),
            (2, 5),
            (((2, 3), torch.float32), ((3, 5), torch.float32)),
        ),
        (
            build_matmul_ikj(),
            (1, 1),
            (((1, 1), torch.float32), ((1, 1), torch.float32)),
        ),
        (
            build_matmul_ikj(),
            (0, 6),
            (((0, 5), torch.float32), ((5, 6), torch.float32)),
        ),
        (
            build_matmul_ikj(torch.float64),
            (4, 6),
            (((4, 5), torch.float64), ((5, 6), torch.float64)),
        ),
        (
            build_ttm(),
            (3, 4, 6),
            (((3, 4, 5), torch.float32), ((5, 6), torch.float32)),
        ),
    ]
    for cin, result_shape, bindings in cells:
        comparison = compare_generated_sources(
            cin,
            result_shape,
            bindings,
            compile_options=options,
        )
        assert comparison.identical, (result_shape, bindings)


@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_reduce_out_keeps_invalid_legacy_dense_position_order_fail_closed(
    regblock_enabled,
):
    """A broader dense auto plan is not evidence for target compatibility.

    Legacy scheduling can choose ``a,b,d,c,e,f`` for this rank-6
    contraction and records a dense reduce-out workspace, but its emitted
    position chain uses ``pA2`` before declaring it.  LoopIR must keep the
    pre-existing ``unsupported_loop_order`` target boundary instead of
    reproducing invalid C++ or letting the reduce-out family gate over-admit
    the program.
    """

    from scorch.compiler.loopir.pipeline import legacy_generated_cpp
    from scorch.compiler.scheduler import Schedule, Scheduler

    a, b, c, d, e, f = (IndexVar(name) for name in "abcdef")
    left = TensorVar("A", fmt="ddddd")
    right = TensorVar("B", fmt="ddd")
    result = TensorVar("C", fmt="dddd")
    cin: IndexStmt = TensorAssign(
        result[a, b, c, f],
        CINBinaryOp(
            Operation.MUL,
            left[a, b, c, d, e],
            right[d, e, f],
        ),
        op=Operation.ADD,
    )
    for index in reversed((a, b, c, d, e, f)):
        cin = ForAll(index, cin)

    plan = Scheduler.auto_schedule_plan(
        cin, regblock_enabled=regblock_enabled
    ).verified_loop_plan
    assert plan.workspace is not None and plan.workspace.dense
    assert len(plan.tiles) == 5

    bindings = (
        ((2, 3, 4, 5, 6), torch.float32),
        ((5, 6, 7), torch.float32),
    )
    options = replace(
        CompileOptions.from_environment(environ={}).with_regblock_enabled(
            regblock_enabled
        ),
        requested_schedule=Schedule(),
    )
    legacy_cpp = legacy_generated_cpp(
        cin,
        (2, 3, 4, 7),
        bindings,
        compile_options=options,
    )
    assert legacy_cpp.index("int pA3 = pA2") < legacy_cpp.index("int pA2 =")
    with pytest.raises(Exception) as error:
        compile_cin_via_loopir(
            cin,
            (2, 3, 4, 7),
            bindings,
            compile_options=options,
        )
    assert getattr(error.value, "defect").code == "unsupported_loop_order"


_SPARSE_WORKSPACE_DIMS = {"i": 4, "j": 5, "k": 6}


def _build_boundary_cin(fmt_result, result_indices, operands, nest):
    ivars = {name: IndexVar(name) for name in "ijk"}
    result = TensorVar("C", fmt=fmt_result)
    rhs = None
    for position, (fmt, indices) in enumerate(operands):
        operand = TensorVar("AB"[position], fmt=fmt)
        access = operand[tuple(ivars[x] for x in indices)]
        rhs = access if rhs is None else CINBinaryOp(Operation.MUL, rhs, access)
    assignment = TensorAssign(
        result[tuple(ivars[x] for x in result_indices)],
        rhs,
        op=Operation.ADD,
    )
    statement = assignment
    for name in reversed(nest):
        statement = ForAll(ivars[name], statement)
    return statement


@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize(
    ("family", "spec", "expected"),
    [
        (
            "spmspm_row_scope",
            ("ds", "ik", (("ss", "ij"), ("ss", "jk")), "ijk"),
            "unsupported_sparse_output_reduction",
        ),
        # The ds@ds->ds SpGEMM cell left this fail-closed census when the
        # Phase-7 parallel sparse-workspace target migrated it; its
        # admission, byte parity, and execution differentials live in
        # test_loopir_parallel_workspace_target.py.
        (
            "reduce_to_csr",
            ("ds", "ik", (("ds", "ij"), ("dd", "jk")), "ijk"),
            "unsupported_sparse_output_reduction",
        ),
        (
            "merged_reduction_dense_out",
            ("dd", "ik", (("ss", "ij"), ("ss", "jk")), "ijk"),
            "unsupported_merged_reduction",
        ),
        (
            "merged_update_dense_out",
            ("dd", "ij", (("ds", "ij"), ("ds", "ij")), "ij"),
            "unsupported_merged_update",
        ),
        (
            "sparse_output_root",
            ("s", "i", (("dd", "ji"), ("d", "j")), "ji"),
            "unsupported_sparse_output",
        ),
        (
            "dense_domain_to_csr",
            ("ds", "ij", (("dd", "ij"),), "ij"),
            "unsupported_sparse_output_domain",
        ),
        (
            "sparse_row_to_csr",
            ("ds", "ij", (("ss", "ij"),), "ij"),
            "unsupported_sparse_output_domain",
        ),
        (
            "mixed_level_dense_axis_reduction",
            ("sd", "ij", (("dd", "ij"),), "ij"),
            None,
        ),
        (
            "mixed_level_merged_operand_chain",
            ("sd", "ij", (("dd", "ij"),), "ij"),
            None,
        ),
    ],
)
def test_sparse_workspace_families_fail_closed_with_exact_codes(
    regblock_enabled, family, spec, expected
):
    """The sparse-result/workspace boundary is exact in both policy arms.

    The B2 mixed dense-leaf assembly family compiles (its battery is
    ``test_loopir_sparse_workspace_target.py``), and the mixed dense-leaf
    OPERAND chain now lowers through declared physical position loads (its
    battery is ``test_loopir_mixed_operand_target.py``); the adjacent seams
    stay fail-closed at precise codes: reducing into a compressed-parent/
    dense-leaf result carries the dense-workspace F2/F4 plan whose mixed
    twin is not migrated (``unsupported_sparse_output_reduction``), and
    merging two mixed dense-leaf operands needs a bound position no merged
    loop provides (``unsupported_sparse_hierarchy``).  True sparse
    ``coo_workspace`` families — row-scope SpMSpM, reduction-to-CSR, merged
    sparse reductions, and sparse-output roots — fail closed at their own
    stable codes, including the early ``unsupported_sparse_output``
    boundary this census audits explicitly.
    """

    from scorch.compiler.scheduler import Schedule

    if family == "mixed_level_dense_axis_reduction":
        i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
        result = TensorVar("C", fmt="sd")
        source = TensorVar("A", fmt="ddd")
        cin = ForAll(
            k,
            ForAll(
                i,
                ForAll(
                    j,
                    TensorAssign(result[k, j], source[k, i, j], op=Operation.ADD),
                ),
            ),
        )
        bindings = (((6, 4, 5), torch.float32),)
        out_shape = (6, 5)
        expected = "unsupported_sparse_output_reduction"
    elif family == "mixed_level_merged_operand_chain":
        i, j = IndexVar("i"), IndexVar("j")
        result = TensorVar("C", fmt="dd")
        source = TensorVar("A", fmt="sd")
        second = TensorVar("B", fmt="sd")
        cin = ForAll(
            i,
            ForAll(
                j,
                TensorAssign(
                    result[i, j],
                    CINBinaryOp(Operation.MUL, source[i, j], second[i, j]),
                ),
            ),
        )
        bindings = (((4, 5), torch.float32), ((4, 5), torch.float32))
        out_shape = (4, 5)
        expected = "unsupported_sparse_hierarchy"
    elif family in ("dense_domain_to_csr", "sparse_row_to_csr"):
        i, j = IndexVar("i"), IndexVar("j")
        result = TensorVar("C", fmt="ds")
        source = TensorVar(
            "A",
            fmt="dd" if family == "dense_domain_to_csr" else "ss",
        )
        cin = ForAll(i, ForAll(j, TensorAssign(result[i, j], source[i, j])))
        bindings = (((4, 5), torch.float32),)
        out_shape = (4, 5)
    else:
        cin = _build_boundary_cin(*spec)
        bindings = tuple(
            (tuple(_SPARSE_WORKSPACE_DIMS[x] for x in indices), torch.float32)
            for _, indices in spec[2]
        )
        out_shape = tuple(_SPARSE_WORKSPACE_DIMS[x] for x in spec[1])
    options = replace(
        CompileOptions.from_environment(environ={}).with_regblock_enabled(
            regblock_enabled
        ),
        requested_schedule=Schedule(),
    )
    with pytest.raises(Exception) as error:
        compile_cin_via_loopir(
            cin,
            out_shape,
            bindings,
            compile_options=options,
        )
    assert getattr(error.value, "defect").code == expected


@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_doubly_compressed_sparse_workspace_compiles_through_the_target(
    regblock_enabled,
):
    """B1 is typed, scheduled, and target-lowered in both automatic arms.

    The completed serial sparse-workspace target owns the automatic
    ``ss@ss->ss`` family end to end: schedule application and LLIR target
    emission are both recorded, and the generated source matches the
    retained legacy ``coo_workspace_1d`` comparand byte for byte.  The
    dedicated differential battery lives in
    ``test_loopir_sparse_workspace_target.py``.
    """

    from scorch.compiler.scheduler import Schedule

    cin = _build_boundary_cin(
        "ss",
        "ij",
        (("ss", "ik"), ("ss", "kj")),
        "ikj",
    )
    options = replace(
        CompileOptions.from_environment(environ={}).with_regblock_enabled(
            regblock_enabled
        ),
        requested_schedule=Schedule(),
    )
    context = CompilationContext(options)
    kernel = compile_cin_via_loopir(
        cin,
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=options,
        compilation_context=context,
    )
    recorded = {record.stage_id for record in context.stage_run_records}
    assert CompilerStageId.LOOPIR_SCHEDULE_APPLICATION in recorded
    assert CompilerStageId.LOOPIR_TO_LLIR_LOWERING in recorded
    assert "coo_workspace_1d<float, 1>(1024)" in kernel.cpp_source
    legacy_cpp = legacy_generated_cpp(
        _build_boundary_cin(
            "ss",
            "ij",
            (("ss", "ik"), ("ss", "kj")),
            "ikj",
        ),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=options,
    )
    assert kernel.cpp_source == legacy_cpp


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


@pytest.mark.parametrize("scheduled", (False, True))
def test_pipeline_preflights_caller_cin_once(monkeypatch, scheduled):
    """Private normalized roots reuse their proven structural boundary."""

    import scorch.compiler.cin_analysis as cin_analysis_module
    from scorch.compiler.scheduler import Schedule

    original = cin_analysis_module._preflight_cin_structure
    calls = 0

    def counted(cin):
        nonlocal calls
        calls += 1
        return original(cin)

    monkeypatch.setattr(
        cin_analysis_module,
        "_preflight_cin_structure",
        counted,
    )
    options = (
        CompileOptions.from_environment(requested_schedule=Schedule())
        if scheduled
        else None
    )

    compile_cin_via_loopir(
        build_elementwise(Operation.ADD),
        (2, 3),
        [((2, 3), torch.float32)] * 2,
        compile_options=options,
    )

    assert calls == 1


def test_source_comparison_ignores_nonsemantic_legacy_metadata():
    """Comparison detaches through normalization, not an unsafe raw deepcopy."""

    class HostileMetadata:
        def __deepcopy__(self, memo):
            raise RuntimeError("nonsemantic metadata was copied")

    cin = build_elementwise(Operation.ADD)
    cin.no_tile_list = [HostileMetadata()]

    comparison = compare_generated_sources(
        cin,
        (2, 3),
        [((2, 3), torch.float32)] * 2,
    )

    assert comparison.identical


def test_pipeline_fails_closed_on_unsupported_unary_expression():
    """Unary CIN reaches the LoopIR boundary without visitor recursion."""

    from scorch.compiler.loopir.lower_cin import LoopIRLoweringError

    i = IndexVar("i")
    operand = TensorVar("A", fmt="d")
    result = TensorVar("C", fmt="d")
    program = ForAll(
        i,
        TensorAssign(result[i], UnaryOp(Operation.SUB, operand[i])),
    )

    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(
            program,
            (3,),
            [((3,), torch.float32)],
        )

    assert error.value.defect.code == "unsupported_expression"


def test_pipeline_root_assignment_reaches_unsupported_statement_boundary():
    """A root result access is not misclassified as a runtime input."""

    from scorch.compiler.loopir.lower_cin import LoopIRLoweringError

    scalar_format = TensorFormat()
    operand = TensorVar("A", fmt=scalar_format)
    result = TensorVar("C", fmt=scalar_format)

    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(
            TensorAssign(result[()], operand[()]),
            (),
            [((), torch.float32)],
        )

    assert error.value.defect.code == "unsupported_statement"


@pytest.mark.parametrize(
    "safe_shape",
    (
        torch.Size((2, 3)),
        range(2, 4),
    ),
)
@pytest.mark.parametrize("entry", (compile_cin_via_loopir, legacy_generated_cpp))
def test_source_entries_preserve_safe_sequence_shapes(entry, safe_shape):
    """Exact torch.Size and range retain the public Sequence shape contract."""

    source = entry(
        build_elementwise(Operation.ADD),
        safe_shape,
        [(safe_shape, torch.float32)] * 2,
    )

    assert source


@pytest.mark.parametrize(
    "safe_shape",
    (
        torch.Size((2, 3)),
        range(2, 4),
    ),
)
def test_direct_execution_preserves_safe_sequence_result_shapes(
    monkeypatch,
    safe_shape,
):
    """The direct public helper accepts safe Sequence shapes before planning."""

    import scorch.ops as ops_module

    class AcceptedShape(Exception):
        pass

    def stop_after_runtime_boundary(*args, **kwargs):
        raise AcceptedShape

    monkeypatch.setattr(
        ops_module,
        "_plan_direct_cin_runtime_binding",
        stop_after_runtime_boundary,
    )
    a = dense_stensor(torch.ones(2, 3), "A")
    b = dense_stensor(torch.ones(2, 3), "B")

    with pytest.raises(AcceptedShape):
        ops_module.lower_and_exec_cin(
            build_elementwise(Operation.ADD),
            safe_shape,
            a,
            b,
        )


@pytest.mark.parametrize(
    "shape",
    (
        range(0, 1 << 100),
        tuple(1 for _ in range(65)),
    ),
)
def test_direct_execution_rejects_unbounded_safe_sequence_rank(shape):
    """Safe built-ins still fail in bounded time above the ABI rank limit."""

    from scorch.ops import lower_and_exec_cin

    with pytest.raises(CompileSpecError, match="rank"):
        lower_and_exec_cin(
            build_elementwise(Operation.ADD),
            shape,
            dense_stensor(torch.ones(2, 3), "A"),
            dense_stensor(torch.ones(2, 3), "B"),
        )


def test_runtime_metadata_late_format_mismatch_is_atomic():
    """Compile-only format validation completes before any tensor metadata bind."""

    from scorch.compiler.loopir.pipeline import _bind_runtime_metadata

    cin = build_elementwise(Operation.ADD)
    rhs = cin.get_rhs_tensor_vars()
    results = cin.get_result_tensor_vars()
    before = tuple(
        (tensor.shape, tensor.dtype, tuple(tensor.mode_order))
        for tensor in (*rhs, *results)
    )

    with pytest.raises(CompileSpecError, match="expects format"):
        _bind_runtime_metadata(
            cin,
            [
                (torch.Size((2, 3)), torch.float32),
                (torch.Size((2, 3)), torch.float32),
            ],
            torch.Size((2, 3)),
            (rhs[0].format, TensorFormat("ds")),
        )

    after = tuple(
        (tensor.shape, tensor.dtype, tuple(tensor.mode_order))
        for tensor in (*rhs, *results)
    )
    assert after == before


@pytest.mark.parametrize(
    "result_shape,input_bindings",
    (
        ((1 << 63,), (((1,), torch.float32),)),
        ((1,), (((1 << 63,), torch.float32),)),
        ((1,), (((1,), object()),)),
        ((1,), (((1,),),)),
    ),
)
@pytest.mark.parametrize("entry", (compile_cin_via_loopir, legacy_generated_cpp))
def test_source_entries_validate_runtime_bindings_before_lowering(
    entry,
    result_shape,
    input_bindings,
):
    """Both source routes own the same bounded shape/dtype contract."""

    i = IndexVar("i")
    operand = TensorVar("A", fmt="d")
    result = TensorVar("C", fmt="d")
    program = ForAll(i, TensorAssign(result[i], operand[i]))

    with pytest.raises(VerificationError):
        entry(program, result_shape, input_bindings)


@pytest.mark.parametrize("scheduled", (False, True))
@pytest.mark.parametrize("list_subclass", (False, True))
def test_legacy_source_rejects_hostile_result_shape_without_iteration(
    scheduled,
    list_subclass,
):
    """Legacy source generation reaches the shared exact-shape boundary."""

    from scorch.compiler.scheduler import Schedule

    reads = 0

    class HostileShape:
        def __iter__(self):
            nonlocal reads
            reads += 1
            raise RuntimeError("hostile result-shape iteration")

    class HostileList(list):
        def __iter__(self):
            nonlocal reads
            reads += 1
            raise RuntimeError("hostile result-shape list")

    i = IndexVar("i")
    operand = TensorVar("A", fmt="d")
    result = TensorVar("C", fmt="d")
    program = ForAll(i, TensorAssign(result[i], operand[i]))
    shape = HostileList([1]) if list_subclass else HostileShape()
    options = CompileOptions.from_environment(
        requested_schedule=Schedule() if scheduled else None
    )

    with pytest.raises(VerificationError):
        legacy_generated_cpp(
            program,
            shape,
            (((1,), torch.float32),),
            compile_options=options,
        )
    assert reads == 0


def test_legacy_execution_rejects_unrepresentable_runtime_shape_before_codegen():
    """The production ABI never renders an out-of-range shape literal."""

    from scorch.ops import lower_and_exec_cin

    i = IndexVar("i")
    operand = TensorVar("A", fmt="d")
    result = TensorVar("C", fmt="d")
    program = ForAll(i, TensorAssign(result[i], operand[i]))

    with pytest.raises(ValueError, match="signed int64"):
        lower_and_exec_cin(
            program,
            (1 << 63,),
            dense_stensor(torch.ones(1), "A"),
        )


def test_legacy_execution_rejects_hostile_result_shape_without_iteration():
    """The direct legacy execution entry never pre-consumes forged shapes."""

    from scorch.ops import lower_and_exec_cin

    reads = 0

    class HostileList(list):
        def __iter__(self):
            nonlocal reads
            reads += 1
            raise RuntimeError("hostile direct result shape")

    i = IndexVar("i")
    operand = TensorVar("A", fmt="d")
    result = TensorVar("C", fmt="d")
    program = ForAll(i, TensorAssign(result[i], operand[i]))

    with pytest.raises(CompileSpecError):
        lower_and_exec_cin(
            program,
            HostileList([1]),
            dense_stensor(torch.ones(1), "A"),
        )
    assert reads == 0


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
    result_decl = next(
        decl
        for decl in lowering.program.tensors
        if decl.symbol == lowering.result_symbol
    )
    logical_shape = [0] * len(result_decl.levels)
    for physical_level, level in enumerate(result_decl.levels):
        logical_shape[level.mode] = result_shape[physical_level]
    return run_program(
        lowering.program,
        inputs,
        {lowering.result_symbol: tuple(logical_shape)},
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


# ---------------------------------------------------------------------------
# Non-identity dense layouts: general logical-coordinate -> physical-position
# lowering.  The activation family is the public-aligned
# einsum("ij,kj->ik", ds, dd) constituent (dense B[k, j] over physical
# mode_order=[1, 0]); the census extends to batched, non-involutive,
# multi-operand, matvec, f64, and zero-extent cells.  Legacy compiles and
# executes every one of these, so byte parity is the gate, with compiled
# and oracle differentials proving the shared semantics independently.

_LAYOUT_DIMS = {"i": 4, "j": 6, "k": 5, "l": 3}


def _build_layout_cin(result_spec, operand_specs, nest, *, op, dtype):
    ivars = {name: IndexVar(name) for name in nest}
    fmt_result, result_indices = result_spec[:2]
    result_mode_order = result_spec[2] if len(result_spec) == 3 else None
    result = TensorVar("C", fmt=fmt_result, dtype=dtype)
    if result_mode_order is not None:
        result.mode_order = list(result_mode_order)
    rhs = None
    names = ["A", "B", "D", "E"]
    for position, (fmt, indices, mode_order) in enumerate(operand_specs):
        operand = TensorVar(names[position], fmt=fmt, dtype=dtype)
        if mode_order is not None:
            operand.mode_order = list(mode_order)
        access = operand[tuple(ivars[x] for x in indices)]
        rhs = access if rhs is None else CINBinaryOp(Operation.MUL, rhs, access)
    assignment = TensorAssign(
        result[tuple(ivars[x] for x in result_indices)], rhs, op=op
    )
    statement = assignment
    for name in reversed(nest):
        statement = ForAll(ivars[name], statement)
    return statement


def _layout_physical_shape(indices, mode_order, dims):
    logical = tuple(dims[x] for x in indices)
    if mode_order is None:
        return logical
    return tuple(logical[mode] for mode in mode_order)


_LAYOUT_FAMILIES = {
    "transposed_spmm": (
        ("dd", "ik"),
        (("ds", "ij", None), ("dd", "kj", (1, 0))),
        "ijk",
        Operation.ADD,
        torch.float32,
        None,
    ),
    "transposed_spmm_f64": (
        ("dd", "ik"),
        (("ds", "ij", None), ("dd", "kj", (1, 0))),
        "ijk",
        Operation.ADD,
        torch.float64,
        None,
    ),
    "transposed_spmm_zero_rows": (
        ("dd", "ik"),
        (("ds", "ij", None), ("dd", "kj", (1, 0))),
        "ijk",
        Operation.ADD,
        torch.float32,
        {"i": 0},
    ),
    "transposed_spmm_zero_reduction": (
        ("dd", "ik"),
        (("ds", "ij", None), ("dd", "kj", (1, 0))),
        "ijk",
        Operation.ADD,
        torch.float32,
        {"j": 0},
    ),
    "transposed_spmm_zero_free": (
        ("dd", "ik"),
        (("ds", "ij", None), ("dd", "kj", (1, 0))),
        "ijk",
        Operation.ADD,
        torch.float32,
        {"k": 0},
    ),
    "batched_permuted": (
        ("ddd", "lik"),
        (("ddd", "lij", None), ("ddd", "lkj", (0, 2, 1))),
        "lijk",
        Operation.ADD,
        torch.float32,
        None,
    ),
    "noninvolutive_elementwise": (
        ("ddd", "ijk"),
        (("ddd", "ijk", None), ("ddd", "kij", (1, 2, 0))),
        "ijk",
        None,
        torch.float32,
        None,
    ),
    "multi_operand_transposed": (
        ("dd", "ik"),
        (("ds", "ij", None), ("dd", "kj", (1, 0)), ("dd", "ik", None)),
        "ijk",
        Operation.ADD,
        torch.float32,
        None,
    ),
    "matvec_transposed": (
        ("d", "i"),
        (("dd", "ji", (1, 0)), ("d", "j", None)),
        "ij",
        Operation.ADD,
        torch.float32,
        None,
    ),
    "permuted_result_matmul": (
        ("dd", "ik", (1, 0)),
        (("dd", "ij", (1, 0)), ("dd", "jk", None)),
        "jki",
        Operation.ADD,
        torch.float32,
        None,
    ),
    "permuted_result_rank3": (
        ("ddd", "ijk", (1, 2, 0)),
        (("ddd", "ijk", (1, 2, 0)),),
        "jki",
        None,
        torch.float32,
        None,
    ),
    "permuted_result_rank3_zero": (
        ("ddd", "ijk", (1, 2, 0)),
        (("ddd", "ijk", (1, 2, 0)),),
        "jki",
        None,
        torch.float32,
        {"i": 0},
    ),
}


@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize("family", sorted(_LAYOUT_FAMILIES))
def test_permuted_dense_layout_byte_parity(family, regblock_enabled):
    """Every admitted permuted-layout cell is byte-identical to legacy."""

    from scorch.compiler.scheduler import Schedule

    result_spec, operand_specs, nest, op, dtype, overrides = _LAYOUT_FAMILIES[family]
    dims = dict(_LAYOUT_DIMS)
    if overrides:
        dims.update(overrides)
    options = replace(
        CompileOptions.from_environment(environ={}).with_regblock_enabled(
            regblock_enabled
        ),
        requested_schedule=Schedule(),
    )
    result_mode_order = result_spec[2] if len(result_spec) == 3 else None
    result_shape = _layout_physical_shape(
        result_spec[1],
        result_mode_order,
        dims,
    )
    bindings = tuple(
        (_layout_physical_shape(indices, mode_order, dims), dtype)
        for _, indices, mode_order in operand_specs
    )
    comparison = compare_generated_sources(
        _build_layout_cin(result_spec, operand_specs, nest, op=op, dtype=dtype),
        result_shape,
        bindings,
        compile_options=options,
    )
    assert comparison.identical, family


@torch.no_grad()
def test_transposed_spmm_execution_matches_reference_both_marshalings():
    """Compiled execution matches PyTorch for both runtime input layouts.

    The runtime binding twin accepts the dense operand either as a logical
    (identity) STensor, which it relayouts to the declared physical order,
    or as an STensor already carrying mode_order=[1, 0]; both must execute
    and match the reference, including every zero-extent class.
    """

    torch.manual_seed(20260727)
    for dims in ((4, 6, 5), (0, 6, 5), (4, 0, 5), (4, 6, 0)):
        rows, reduction, free = dims
        a_t = (torch.rand(rows, reduction) * (torch.rand(rows, reduction) > 0.4)).to(
            torch.float32
        )
        b_t = torch.rand(free, reduction)
        reference = a_t @ b_t.T
        sparse = STensor.from_torch(a_t.to_sparse_csr(), "A")
        for variant_input in (
            STensor.from_torch(b_t, "B"),
            STensor.from_torch(b_t, "B", mode_order=[1, 0]).to_dense(),
        ):
            result, _ = execute_cin_via_loopir(
                _build_layout_cin(
                    ("dd", "ik"),
                    (("ds", "ij", None), ("dd", "kj", (1, 0))),
                    "ijk",
                    op=Operation.ADD,
                    dtype=torch.float32,
                ),
                (rows, free),
                sparse,
                variant_input,
            )
            assert_close(result.to_torch(), reference)


@torch.no_grad()
def test_permuted_layout_execution_and_oracle_agree():
    """Batched and non-involutive permutations execute and match the oracle.

    Byte parity alone could silently reproduce defective legacy output, so
    the compiled kernels must independently match PyTorch, and the
    production oracle (which consumes logical nested lists) must agree on
    the transposed activation program.
    """

    from scorch.compiler.scheduler import Schedule

    torch.manual_seed(20260728)
    batch, rows, reduction, free = 3, 4, 6, 5

    a_t = torch.rand(batch, rows, reduction)
    b_t = torch.rand(batch, free, reduction)
    result, _ = execute_cin_via_loopir(
        _build_layout_cin(
            ("ddd", "lik"),
            (("ddd", "lij", None), ("ddd", "lkj", (0, 2, 1))),
            "lijk",
            op=Operation.ADD,
            dtype=torch.float32,
        ),
        (batch, rows, free),
        dense_stensor(a_t, "A"),
        dense_stensor(b_t, "B"),
    )
    assert_close(result.to_torch(), torch.einsum("lij,lkj->lik", a_t, b_t))

    a3_t = torch.rand(rows, reduction, free)
    b3_t = torch.rand(free, rows, reduction)
    result, _ = execute_cin_via_loopir(
        _build_layout_cin(
            ("ddd", "ijk"),
            (("ddd", "ijk", None), ("ddd", "kij", (1, 2, 0))),
            "ijk",
            op=None,
            dtype=torch.float32,
        ),
        (rows, reduction, free),
        dense_stensor(a3_t, "A"),
        dense_stensor(b3_t, "B"),
    )
    assert_close(result.to_torch(), a3_t * b3_t.permute(1, 2, 0))

    # The public all-dense matmul frontend uses this temporary result layout:
    # every tensor follows the common j,k,i storage order, while the returned
    # tensor still exposes logical C[i,k].
    a2_t = torch.rand(rows, reduction)
    b2_t = torch.rand(reduction, free)
    result, kernel = execute_cin_via_loopir(
        _build_layout_cin(
            ("dd", "ik", (1, 0)),
            (("dd", "ij", (1, 0)), ("dd", "jk", None)),
            "jki",
            op=Operation.ADD,
            dtype=torch.float32,
        ),
        (free, rows),
        dense_stensor(a2_t, "A"),
        dense_stensor(b2_t, "B"),
    )
    assert tuple(result.mode_order) == (1, 0)
    assert result.shape == (free, rows)
    assert_close(result.to_torch(), a2_t @ b2_t)
    assert_close(
        oracle_reference(kernel, (a2_t, b2_t), (free, rows)),
        (a2_t @ b2_t).to(torch.float64),
    )

    a3_result_t = torch.rand(rows, reduction, free)
    result, kernel = execute_cin_via_loopir(
        _build_layout_cin(
            ("ddd", "ijk", (1, 2, 0)),
            (("ddd", "ijk", (1, 2, 0)),),
            "jki",
            op=None,
            dtype=torch.float32,
        ),
        (reduction, free, rows),
        dense_stensor(a3_result_t, "A"),
    )
    assert tuple(result.mode_order) == (1, 2, 0)
    assert result.shape == (reduction, free, rows)
    assert_close(result.to_torch(), a3_result_t)
    assert_close(
        oracle_reference(
            kernel,
            (a3_result_t,),
            (reduction, free, rows),
        ),
        a3_result_t.to(torch.float64),
    )

    empty_result_t = torch.empty(0, reduction, free)
    result, _ = execute_cin_via_loopir(
        _build_layout_cin(
            ("ddd", "ijk", (1, 2, 0)),
            (("ddd", "ijk", (1, 2, 0)),),
            "jki",
            op=None,
            dtype=torch.float32,
        ),
        (reduction, free, 0),
        dense_stensor(empty_result_t, "A"),
    )
    assert tuple(result.mode_order) == (1, 2, 0)
    assert result.shape == (reduction, free, 0)
    assert_close(result.to_torch(), empty_result_t)

    a_dense = torch.rand(rows, reduction)
    b_dense = torch.rand(free, reduction)
    kernel = compile_cin_via_loopir(
        _build_layout_cin(
            ("dd", "ik"),
            (("dd", "ij", None), ("dd", "kj", (1, 0))),
            "ijk",
            op=Operation.ADD,
            dtype=torch.float32,
        ),
        (rows, free),
        (((rows, reduction), torch.float32), ((reduction, free), torch.float32)),
        compile_options=replace(
            CompileOptions.from_environment(environ={}),
            requested_schedule=Schedule(),
        ),
    )
    oracle_result = oracle_reference(kernel, (a_dense, b_dense), (rows, free))
    assert_close(oracle_result.to(torch.float32), a_dense @ b_dense.T)


@torch.no_grad()
def test_unscheduled_shadow_preserves_permuted_dense_result_layout():
    """The advanced legacy entry wraps physical result storage logically."""

    logical = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    carried = STensor.from_torch(
        logical,
        "A",
        mode_order=[1, 0],
    ).to_dense()
    cin = _build_layout_cin(
        ("dd", "ij", (1, 0)),
        (("dd", "ij", (1, 0)),),
        "ji",
        op=None,
        dtype=torch.float32,
    )

    loopir_result, legacy_result, comparison = execute_shadow(
        cin,
        (3, 2),
        carried,
    )

    assert comparison.identical
    assert tuple(loopir_result.mode_order) == (1, 0)
    assert tuple(legacy_result.mode_order) == (1, 0)
    assert loopir_result.shape == (3, 2)
    assert legacy_result.shape == (3, 2)
    assert torch.equal(loopir_result.to_torch(), logical)
    assert torch.equal(legacy_result.to_torch(), logical)
