"""B1 serial sparse-workspace target: byte parity and honest sparse output.

The completed LoopIR target for the automatic ``ss@ss->ss`` family must
generate byte-identical C++ to the retained serial ``coo_workspace_1d``
legacy pipeline in both automatic policy arms, execute through the shared
JIT build path, and return an honest doubly-compressed ``STensor`` whose
``pos``/``crd``/value storage matches the production LoopIR oracle exactly
(including explicit zeros) and the PyTorch dense reference within the
repository tolerance.  The dense-only shadow helper is deliberately not
used anywhere here: sparse results are validated at the storage level.
"""

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
    CompilerStageId,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_llir import (
    LoopIRTargetError,
    lower_loopir_to_llir,
)
from scorch.compiler.loopir.nodes import (
    BinaryOp as LoopIRBinaryOp,
    LevelKind,
    MergeMode,
    ReduceOp,
    ScalarType,
)
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compare_generated_sources,
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    legacy_generated_cpp,
)
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.schedule_passes import verify_scheduled_loopir
from scorch.compiler.scheduler import Schedule
from scorch.stensor import STensor

_CC_KINDS = (LevelKind.COMPRESSED, LevelKind.COMPRESSED)


def build_spmspm_cin(
    dtype=torch.float32,
    *,
    commuted=False,
    index_names=("i", "k", "j"),
):
    i, k, j = (IndexVar(name) for name in index_names)
    a = TensorVar("A", fmt="ss", dtype=dtype)
    b = TensorVar("B", fmt="ss", dtype=dtype)
    c = TensorVar("C", fmt="ss", dtype=dtype)
    value = (
        CINBinaryOp(Operation.MUL, b[k, j], a[i, k])
        if commuted
        else CINBinaryOp(Operation.MUL, a[i, k], b[k, j])
    )
    assign = TensorAssign(
        c[i, j],
        value,
        op=Operation.ADD,
    )
    return ForAll(i, ForAll(k, ForAll(j, assign)))


def auto_options(regblock_enabled, *, jit=False):
    """The automatic-policy compile options for one arm.

    Compile-only tests pin ``environ={}``; executing tests need the real
    environment because the JIT build snapshot owns the toolchain PATH.
    """

    base = (
        CompileOptions.from_environment()
        if jit
        else CompileOptions.from_environment(environ={})
    )
    return replace(
        base.with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(),
    )


def sparse_ss(dense, name):
    return STensor.from_torch(dense.clone(), name).to_sparse("ss")


def validated_ss_storage(result, shape):
    """Assert honest identity-ordered ``ss`` storage; return its pieces."""

    assert str(result.index.format) == "s,s"
    assert tuple(result.index.mode_order) == (0, 1)
    assert tuple(result.shape) == tuple(shape)
    mode_indices = result.index.mode_indices
    assert len(mode_indices) == 2
    assert all(len(level_pair) == 2 for level_pair in mode_indices)
    pos0 = mode_indices[0][0].tolist()
    crd0 = mode_indices[0][1].tolist()
    pos1 = mode_indices[1][0].tolist()
    crd1 = mode_indices[1][1].tolist()
    values = result.values.tolist()
    assert pos0 == [0, len(crd0)]
    assert all(
        0 <= row < shape[0] for row in crd0
    ), f"row coordinate escapes extent {shape[0]}"
    assert crd0 == sorted(set(crd0)), "row coordinates must strictly increase"
    assert len(pos1) == len(crd0) + 1
    assert pos1[0] == 0
    assert pos1 == sorted(pos1), "column positions must be nondecreasing"
    assert pos1[-1] == len(crd1) == len(values)
    for segment in range(len(crd0)):
        columns = crd1[pos1[segment] : pos1[segment + 1]]
        assert columns == sorted(
            set(columns)
        ), "column coordinates must strictly increase per row segment"
        assert all(0 <= column < shape[1] for column in columns)
    return pos0, crd0, pos1, crd1, values


def dense_from_storage(storage, shape, dtype):
    _, crd0, pos1, crd1, values = storage
    dense = torch.zeros(shape, dtype=dtype)
    for segment, row in enumerate(crd0):
        for position in range(pos1[segment], pos1[segment + 1]):
            dense[row, crd1[position]] = values[position]
    return dense


def oracle_result(kernel, dense_a, dense_b, result_shape):
    """Run the scheduled program through the production LoopIR oracle."""

    lowering = kernel.lowering
    assert kernel.schedule is not None
    inputs = {}
    for symbol, dense in zip(lowering.rhs_access_symbols, (dense_a, dense_b)):
        inputs[symbol] = LevelTensorStorage.from_dense(
            dense.tolist(), tuple(dense.shape), (0, 1), _CC_KINDS
        )
    outputs = run_program(
        kernel.schedule.program,
        inputs,
        {lowering.result_symbol: tuple(result_shape)},
    )
    base_outputs = run_program(
        lowering.program,
        inputs,
        {lowering.result_symbol: tuple(result_shape)},
    )
    scheduled = outputs[lowering.result_symbol]
    assert scheduled == base_outputs[lowering.result_symbol]
    return scheduled


def assert_storage_matches_oracle(storage, oracle):
    pos0, crd0, pos1, crd1, values = storage
    assert tuple(pos0) == oracle.positions[0]
    assert tuple(crd0) == oracle.coordinates[0]
    assert tuple(pos1) == oracle.positions[1]
    assert tuple(crd1) == oracle.coordinates[1]
    assert len(values) == len(oracle.values)
    for got, expected in zip(values, oracle.values):
        assert got == pytest.approx(expected, abs=1e-5, rel=1e-5)


def execute_b1(cin, result_shape, st_a, st_b, regblock_enabled):
    return execute_cin_via_loopir(
        cin,
        result_shape,
        st_a,
        st_b,
        compile_options=auto_options(regblock_enabled, jit=True),
    )


def execute_legacy_module(cpp_source, result_shape, st_a, st_b, options):
    """Safe legacy execution: run the byte-identical legacy kernel raw."""

    from scorch.ops import (
        _load_validated_prepared_kernel,
        _prepare_generated_kernel_build,
        _snapshot_runtime_tensors,
    )

    context = CompilationContext(options)
    prepared = _prepare_generated_kernel_build(
        options.build.preamble_source,
        cpp_source,
        options,
        context,
    )
    module = _load_validated_prepared_kernel(prepared)
    snapshots = _snapshot_runtime_tensors((st_a, st_b), expected_count=2)
    module_args = [tuple(result_shape)]
    for snapshot in snapshots:
        module_args.append(snapshot.shape)
        module_args.append(snapshot.native_mode_indices)
        module_args.append(snapshot.values)
    return module.evaluate(*module_args)


# -- source parity ----------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_source_parity_matches_legacy_in_both_arms(dtype):
    sources = {}
    for regblock_enabled in (False, True):
        comparison = compare_generated_sources(
            build_spmspm_cin(dtype),
            (4, 5),
            (((4, 6), dtype), ((6, 5), dtype)),
            compile_options=auto_options(regblock_enabled),
        )
        assert (
            comparison.identical
        ), f"regblock={regblock_enabled} LoopIR C++ diverges from legacy"
        sources[regblock_enabled] = comparison.loopir_cpp
    assert sources[False] == sources[True], "the two policy arms must agree"
    element = "float" if dtype is torch.float32 else "double"
    assert f"coo_workspace_1d<{element}, 1>(1024)" in sources[False]
    assert "wksp.sort();" in sources[False]
    assert "C1_crd.emplace_back(j);" in sources[False]
    assert "C0_crd.push_back(i);" in sources[False]
    assert "scorch_vector_set(C0_pos, C0_pos_index + 1, C0_crd.size());" in (
        sources[False]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_commuted_rhs_cursor_order_still_matches_legacy(regblock_enabled, dtype):
    """The workspace pass does not promise an operand-position ordering."""

    comparison = compare_generated_sources(
        build_spmspm_cin(dtype, commuted=True),
        (4, 5),
        # RHS discovery follows the source expression: B, then A.
        (((6, 5), dtype), ((4, 6), dtype)),
        compile_options=auto_options(regblock_enabled),
    )
    assert comparison.identical


@torch.no_grad()
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_commuted_rhs_cursor_order_executes_against_oracle_and_pytorch(
    regblock_enabled,
):
    torch.manual_seed(20260801)
    dense_a = (torch.rand(4, 6) < 0.5) * torch.randn(4, 6)
    dense_b = (torch.rand(6, 5) < 0.5) * torch.randn(6, 5)
    result, kernel = execute_cin_via_loopir(
        build_spmspm_cin(commuted=True),
        (4, 5),
        # The module ABI follows the commuted RHS discovery order.
        sparse_ss(dense_b, "B"),
        sparse_ss(dense_a, "A"),
        compile_options=auto_options(regblock_enabled, jit=True),
    )
    storage = validated_ss_storage(result, (4, 5))
    dense_result = dense_from_storage(storage, (4, 5), torch.float32)
    assert torch.allclose(dense_result, dense_a @ dense_b, atol=1e-3, rtol=1e-3)
    assert_storage_matches_oracle(
        storage,
        oracle_result(kernel, dense_b, dense_a, (4, 5)),
    )


@pytest.mark.parametrize(
    "reserved_name",
    ["coo_workspace_1d", "int64_t", "size_t"],
)
def test_runtime_and_type_names_cannot_shadow_b1_emission(reserved_name):
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spmspm_cin(
                index_names=(reserved_name, "reduction", "column"),
            ),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert error.value.defect.code == "generated_name_collision"


def test_sparse_workspace_capacity_is_an_exact_integral_literal(monkeypatch):
    """The typed INT literal must not hide a legacy string primitive."""

    from scorch.compiler.loopir import lower_llir as target

    original = target._SparseWorkspaceLowering._region_statements
    observed = []

    def inspect_region(lowering):
        statements = original(lowering)
        workspace_init = statements[1]
        assert type(workspace_init) is target.llir.VarInit
        workspace_call = workspace_init.value
        assert type(workspace_call) is target.llir.FunctionCall
        capacity = workspace_call.args[0]
        observed.append(
            (
                type(capacity),
                type(capacity.value),
                capacity.value,
                capacity.data_type,
            )
        )
        return statements

    monkeypatch.setattr(
        target._SparseWorkspaceLowering,
        "_region_statements",
        inspect_region,
    )
    compile_cin_via_loopir(
        build_spmspm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    assert observed == [(target.llir.Literal, int, 1024, target.llir.DataType.INT)]


def test_sparse_workspace_fails_closed_if_vector_rewrite_is_lost(monkeypatch):
    import scorch.compiler.llir_pass_manager as pass_manager

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        lambda value, _context: value,
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spmspm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert error.value.defect.code == "sparse_workspace_completion_lost"


@pytest.mark.parametrize("tamper", ["metadata_duplicate", "stray_append"])
def test_sparse_workspace_census_rejects_hidden_duplicate_effects(
    monkeypatch,
    tamper,
):
    """Non-rendered metadata cannot hide an extra workspace drain or write."""

    from copy import deepcopy

    import scorch.compiler.llir_pass_manager as pass_manager
    from scorch.compiler import llir
    from scorch.compiler.llir_traversal import (
        LLIRRewriter,
        LLIRStatementSequence,
        LLIRTraversalContext,
    )

    original = pass_manager.rewrite_dynamic_vector_accesses

    class DuplicateDrain(LLIRRewriter):
        def __init__(self):
            super().__init__(
                LLIRTraversalContext(
                    stage="test",
                    pass_name="duplicate_sparse_workspace_drain",
                )
            )
            self.done = False

        def prepare_statement_sequence(
            self,
            statements: LLIRStatementSequence,
            path,
        ):
            prepared = list(statements)
            if self.done:
                return prepared
            for index, statement in enumerate(prepared):
                if (
                    type(statement) is llir.ForLoopAuto
                    and type(statement.array) is llir.Var
                    and statement.array.name == "wksp"
                ):
                    duplicate = deepcopy(statement)
                    duplicate.array.type = llir.DataType.NO_TYPE
                    prepared.insert(index + 1, duplicate)
                    self.done = True
                    break
            return prepared

    def tamper_with_rewrite(value, context):
        rewritten = original(value, context)
        if tamper == "metadata_duplicate":
            duplicate_rewriter = DuplicateDrain()
            result = duplicate_rewriter.rewrite(rewritten)
            assert duplicate_rewriter.done
            return result
        assert type(rewritten) is list
        rewritten.append(
            llir.FunctionCallStmt(
                name="C_values.emplace_back",
                args=[llir.Literal(0.0)],
            )
        )
        return rewritten

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        tamper_with_rewrite,
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spmspm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(False),
        )
    assert error.value.defect.code == "sparse_workspace_completion_lost"


@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_completed_target_owns_the_pipeline_route(regblock_enabled):
    """B1 compiles end to end: schedule applied, target emission recorded."""

    options = auto_options(regblock_enabled)
    context = CompilationContext(options)
    kernel = compile_cin_via_loopir(
        build_spmspm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=options,
        compilation_context=context,
    )
    stages = {record.stage_id for record in context.stage_run_records}
    assert CompilerStageId.CIN_TO_LOOPIR_LOWERING in stages
    assert CompilerStageId.LOOPIR_SCHEDULE_APPLICATION in stages
    assert CompilerStageId.LOOPIR_TO_LLIR_LOWERING in stages
    assert kernel.schedule is not None
    verify_scheduled_loopir(kernel.schedule)
    assert '"kind":"sparse_workspace_region"' in kernel.program_dump
    assert kernel.cpp_source == legacy_generated_cpp(
        build_spmspm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(regblock_enabled),
    )

    replay = compile_cin_via_loopir(
        build_spmspm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(regblock_enabled),
    )
    assert replay.cpp_source == kernel.cpp_source
    assert replay.request_identity == kernel.request_identity
    assert replay.program_dump == kernel.program_dump


def test_unscheduled_route_keeps_its_boundary():
    """Without the automatic plan the semantic base program stays rejected."""

    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            build_spmspm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=options,
        )
    assert error.value.defect.code == "unsupported_program_shape"


# -- compiled sparse-output execution ---------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_execution_matches_every_reference(regblock_enabled, dtype):
    torch.manual_seed(20260728)
    dense_a = torch.zeros((5, 6), dtype=dtype)
    dense_b = torch.zeros((6, 4), dtype=dtype)
    mask_a = torch.rand(5, 6) < 0.45
    mask_b = torch.rand(6, 4) < 0.55
    dense_a[mask_a] = torch.randn(int(mask_a.sum()), dtype=dtype)
    dense_b[mask_b] = torch.randn(int(mask_b.sum()), dtype=dtype)

    result, kernel = execute_b1(
        build_spmspm_cin(dtype),
        (5, 4),
        sparse_ss(dense_a, "A"),
        sparse_ss(dense_b, "B"),
        regblock_enabled,
    )
    storage = validated_ss_storage(result, (5, 4))
    dense_result = dense_from_storage(storage, (5, 4), dtype)
    expected = dense_a @ dense_b
    tolerance = 1e-3
    assert torch.allclose(dense_result, expected, atol=tolerance, rtol=tolerance)
    assert_storage_matches_oracle(
        storage, oracle_result(kernel, dense_a, dense_b, (5, 4))
    )

    # Safe legacy execution: independently generate the exact request's
    # legacy source, prove parity, then compile that source rather than the
    # candidate a second time.
    legacy_cpp = legacy_generated_cpp(
        build_spmspm_cin(dtype),
        (5, 4),
        (((5, 6), dtype), ((6, 4), dtype)),
        compile_options=auto_options(regblock_enabled),
    )
    assert legacy_cpp == kernel.cpp_source
    legacy_raw = execute_legacy_module(
        legacy_cpp,
        (5, 4),
        sparse_ss(dense_a, "A"),
        sparse_ss(dense_b, "B"),
        auto_options(regblock_enabled, jit=True),
    )
    legacy_modes = legacy_raw.storage.index.mode_indices
    assert legacy_modes[0][0].tolist() == storage[0]
    assert legacy_modes[0][1].tolist() == storage[1]
    assert legacy_modes[1][0].tolist() == storage[2]
    assert legacy_modes[1][1].tolist() == storage[3]
    assert legacy_raw.storage.value.tolist() == pytest.approx(storage[4])

    # Independent production execution through the public dispatch.
    import scorch

    public = scorch.matmul(sparse_ss(dense_a, "A"), sparse_ss(dense_b, "B"))
    public_dense = public.to_torch().to(dtype)
    assert torch.allclose(public_dense, expected, atol=tolerance, rtol=tolerance)


@torch.no_grad()
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_empty_inputs_rows_and_disjoint_supports(regblock_enabled):
    dtype = torch.float32
    cells = {
        "both_empty": (torch.zeros((5, 6)), torch.zeros((6, 4))),
        "left_empty": (torch.zeros((5, 6)), torch.ones((6, 4))),
        "right_empty": (torch.ones((5, 6)), torch.zeros((6, 4))),
    }
    disjoint_a = torch.zeros((5, 6))
    disjoint_a[:, :3] = torch.rand(5, 3)
    disjoint_b = torch.zeros((6, 4))
    disjoint_b[3:, :] = torch.rand(3, 4)
    cells["disjoint_reduction_support"] = (disjoint_a, disjoint_b)
    ragged_a = torch.zeros((5, 6))
    ragged_a[0, :] = torch.rand(6)
    ragged_a[2, 4] = 3.0
    ragged_a[4, 0] = -1.0
    ragged_b = torch.zeros((6, 4))
    ragged_b[0, 1] = 2.0
    ragged_b[4, :] = torch.rand(4)
    ragged_b[5, 3] = -2.0
    cells["ragged_rows"] = (ragged_a, ragged_b)

    for name, (dense_a, dense_b) in cells.items():
        result, kernel = execute_b1(
            build_spmspm_cin(dtype),
            (5, 4),
            sparse_ss(dense_a, "A"),
            sparse_ss(dense_b, "B"),
            regblock_enabled,
        )
        storage = validated_ss_storage(result, (5, 4))
        dense_result = dense_from_storage(storage, (5, 4), dtype)
        expected = dense_a @ dense_b
        assert torch.allclose(dense_result, expected, atol=1e-3, rtol=1e-3), name
        assert_storage_matches_oracle(
            storage, oracle_result(kernel, dense_a, dense_b, (5, 4))
        )
        if name in ("both_empty", "left_empty", "right_empty"):
            assert storage[4] == [], f"{name} must produce an empty result"


@torch.no_grad()
@pytest.mark.parametrize(
    ("a_shape", "b_shape"),
    [((0, 6), (6, 4)), ((5, 0), (0, 4)), ((5, 6), (6, 0))],
)
def test_zero_extent_cells(a_shape, b_shape):
    dtype = torch.float32
    dense_a = torch.zeros(a_shape)
    dense_b = torch.zeros(b_shape)
    if dense_a.numel():
        dense_a[0, 0] = 1.0
    if dense_b.numel():
        dense_b[0, 0] = 1.0
    result_shape = (a_shape[0], b_shape[1])
    result, kernel = execute_b1(
        build_spmspm_cin(dtype),
        result_shape,
        sparse_ss(dense_a, "A"),
        sparse_ss(dense_b, "B"),
        False,
    )
    storage = validated_ss_storage(result, result_shape)
    assert storage[4] == []
    assert_storage_matches_oracle(
        storage, oracle_result(kernel, dense_a, dense_b, result_shape)
    )


@torch.no_grad()
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_cancellation_retains_explicit_zeros(regblock_enabled):
    """Exact cancellation keeps the coordinate with an explicit zero value."""

    dtype = torch.float32
    dense_a = torch.zeros((5, 6))
    dense_b = torch.zeros((6, 4))
    dense_a[1, 2] = 1.0
    dense_a[1, 3] = -1.0
    dense_b[2, 0] = 4.0
    dense_b[3, 0] = 4.0
    # And one non-cancelling entry in another row.
    dense_a[3, 0] = 2.0
    dense_b[0, 2] = 1.5

    result, kernel = execute_b1(
        build_spmspm_cin(dtype),
        (5, 4),
        sparse_ss(dense_a, "A"),
        sparse_ss(dense_b, "B"),
        regblock_enabled,
    )
    storage = validated_ss_storage(result, (5, 4))
    pos0, crd0, pos1, crd1, values = storage
    assert crd0 == [1, 3]
    assert 0.0 in values, "the cancelled coordinate must retain an explicit zero"
    assert_storage_matches_oracle(
        storage, oracle_result(kernel, dense_a, dense_b, (5, 4))
    )


@torch.no_grad()
def test_deterministic_drain_ordering_and_replay():
    dtype = torch.float32
    torch.manual_seed(20260729)
    dense_a = (torch.rand(5, 6) < 0.6).to(dtype) * torch.randn(5, 6)
    dense_b = (torch.rand(6, 4) < 0.6).to(dtype) * torch.randn(6, 4)
    first, _ = execute_b1(
        build_spmspm_cin(dtype),
        (5, 4),
        sparse_ss(dense_a, "A"),
        sparse_ss(dense_b, "B"),
        False,
    )
    second, _ = execute_b1(
        build_spmspm_cin(dtype),
        (5, 4),
        sparse_ss(dense_a, "A"),
        sparse_ss(dense_b, "B"),
        False,
    )
    first_storage = validated_ss_storage(first, (5, 4))
    second_storage = validated_ss_storage(second, (5, 4))
    assert first_storage == second_storage


# -- adversarial target boundaries ------------------------------------------


def build_dense_outer_region_program():
    """A verified 1-D dense-outer sparse-workspace program: not the family."""

    builder = LoopIRBuilder()
    dimension = builder.dimension("i")
    output = builder.new_symbol_id()
    output_decl = builder.tensor(
        output,
        "C",
        ScalarType.FLOAT32,
        (dimension.dimension,),
        (builder.level(LevelKind.COMPRESSED, 0),),
    )
    outer_index = builder.new_index_id()
    workspace = builder.new_workspace_id()
    workspace_decl = builder.sparse_workspace_decl(
        workspace, "wksp", ScalarType.FLOAT32, dimension.dimension
    )
    insert = builder.sparse_workspace_insert(
        workspace,
        builder.index_value(outer_index),
        ReduceOp.ADD,
        builder.float_const(1.0),
    )
    drain_index = builder.new_index_id()
    append = builder.append_entry(
        output,
        (builder.index_value(drain_index),),
        builder.sparse_workspace_value(workspace),
    )
    drain = builder.sparse_workspace_drain_for(
        workspace, drain_index, builder.block((append,))
    )
    region = builder.sparse_workspace_region(
        workspace_decl, builder.block((insert,)), builder.block((drain,))
    )
    return builder.program(
        (dimension,),
        (output_decl,),
        (),
        (output,),
        builder.block(
            (
                builder.dense_for(
                    outer_index, dimension.dimension, builder.block((region,))
                ),
            )
        ),
    )


def build_b1_program(insert_value_builder=None):
    """Hand-build the exact B1 chain; optionally forge the insert value."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_k = builder.dimension("k")
    dim_j = builder.dimension("j")
    symbol_a = builder.new_symbol_id()
    symbol_b = builder.new_symbol_id()
    symbol_c = builder.new_symbol_id()

    def cc_tensor(symbol, name, dims):
        return builder.tensor(
            symbol,
            name,
            ScalarType.FLOAT32,
            dims,
            (
                builder.level(LevelKind.COMPRESSED, 0),
                builder.level(LevelKind.COMPRESSED, 1),
            ),
        )

    decl_a = cc_tensor(symbol_a, "A", (dim_i.dimension, dim_k.dimension))
    decl_b = cc_tensor(symbol_b, "B", (dim_k.dimension, dim_j.dimension))
    decl_c = cc_tensor(symbol_c, "C", (dim_i.dimension, dim_j.dimension))

    index_i = builder.new_index_id()
    index_k = builder.new_index_id()
    index_j = builder.new_index_id()
    position_a0 = builder.new_position_id()
    position_a1 = builder.new_position_id()
    position_b0 = builder.new_position_id()
    position_b1 = builder.new_position_id()
    cursor_a0 = builder.sparse_cursor(
        builder.new_cursor_id(), symbol_a, 0, builder.root_position()
    )
    cursor_a1 = builder.sparse_cursor(
        builder.new_cursor_id(), symbol_a, 1, builder.position_value(position_a0)
    )
    cursor_b0 = builder.sparse_cursor(
        builder.new_cursor_id(), symbol_b, 0, builder.root_position()
    )
    cursor_b1 = builder.sparse_cursor(
        builder.new_cursor_id(), symbol_b, 1, builder.position_value(position_b0)
    )
    workspace = builder.new_workspace_id()
    workspace_decl = builder.sparse_workspace_decl(
        workspace, "wksp", ScalarType.FLOAT32, dim_j.dimension
    )
    if insert_value_builder is None:
        insert_value = builder.binary(
            LoopIRBinaryOp.MUL,
            builder.cursor_value(cursor_a1.cursor),
            builder.cursor_value(cursor_b1.cursor),
        )
    else:
        insert_value = insert_value_builder(builder, cursor_a1, cursor_b1)
    insert = builder.sparse_workspace_insert(
        workspace,
        builder.index_value(index_j),
        ReduceOp.ADD,
        insert_value,
    )
    child = builder.sparse_for(
        cursor_b1, position_b1, index_j, builder.block((insert,))
    )
    merge = builder.merged_sparse_for(
        MergeMode.INTERSECTION,
        (cursor_a1, cursor_b0),
        index_k,
        builder.block((child,)),
        positions=(position_a1, position_b0),
    )
    drain_index = builder.new_index_id()
    append = builder.append_entry(
        symbol_c,
        (builder.index_value(index_i), builder.index_value(drain_index)),
        builder.sparse_workspace_value(workspace),
    )
    drain = builder.sparse_workspace_drain_for(
        workspace, drain_index, builder.block((append,))
    )
    region = builder.sparse_workspace_region(
        workspace_decl,
        builder.block((merge,)),
        builder.block((drain,)),
    )
    outer = builder.sparse_for(
        cursor_a0, position_a0, index_i, builder.block((region,))
    )
    return builder.program(
        (dim_i, dim_k, dim_j),
        (decl_a, decl_b, decl_c),
        (symbol_a, symbol_b),
        (symbol_c,),
        builder.block((outer,)),
    ), (symbol_a, symbol_b, symbol_c)


def b1_shapes(symbols):
    symbol_a, symbol_b, _ = symbols
    return {symbol_a: (4, 6), symbol_b: (6, 5)}


def test_hand_built_b1_program_lowers_and_matches_pipeline_source():
    program, symbols = build_b1_program()
    function = lower_loopir_to_llir(
        program,
        input_shapes=b1_shapes(symbols),
        result_shape=(4, 5),
        compile_options=CompileOptions.from_environment(environ={}),
    )
    from scorch.ops import _lower_generated_llir

    options = CompileOptions.from_environment(environ={})
    cpp = _lower_generated_llir(function, options, CompilationContext(options))
    kernel = compile_cin_via_loopir(
        build_spmspm_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    assert cpp == kernel.cpp_source


def test_dense_outer_region_program_stays_outside_the_family():
    program = build_dense_outer_region_program()
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes={},
            result_shape=(4,),
            compile_options=CompileOptions.from_environment(environ={}),
        )
    assert error.value.defect.code == "unsupported_program_shape"


def test_forged_insert_value_fails_closed():
    program, symbols = build_b1_program(
        insert_value_builder=lambda builder, *_: builder.float_const(2.0)
    )
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes=b1_shapes(symbols),
            result_shape=(4, 5),
            compile_options=CompileOptions.from_environment(environ={}),
        )
    assert error.value.defect.code == "unsupported_program_shape"


def test_target_failure_is_a_recorded_stage_loss():
    program = build_dense_outer_region_program()
    options = CompileOptions.from_environment(environ={})
    context = CompilationContext(options)
    with pytest.raises(LoopIRTargetError):
        lower_loopir_to_llir(
            program,
            input_shapes={},
            result_shape=(4,),
            compile_options=options,
            compilation_context=context,
        )
    assert context._failed_stage_id is CompilerStageId.LOOPIR_TO_LLIR_LOWERING
    assert CompilerStageId.LOOPIR_TO_LLIR_LOWERING not in {
        record.stage_id for record in context.stage_run_records
    }


def test_canonical_dump_stability_across_arms():
    dumps = set()
    for regblock_enabled in (False, True):
        kernel = compile_cin_via_loopir(
            build_spmspm_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(regblock_enabled),
        )
        assert kernel.schedule is not None
        dumps.add(canonical_program_dump(kernel.schedule.program))
    assert len(dumps) == 1


# -- B2: the mixed compressed-parent/dense-leaf assembly family --------------


def build_mixed_leaf_cin(rank3=False, dtype=torch.float32, binary=False):
    """Dense-domain elementwise CIN assembling an ``sd``/``sdd`` result."""

    if rank3:
        i, j, m = IndexVar("i"), IndexVar("j"), IndexVar("m")
        result = TensorVar("C", fmt="sdd", dtype=dtype)
        source = TensorVar("A", fmt="ddd", dtype=dtype)
        return ForAll(
            i,
            ForAll(j, ForAll(m, TensorAssign(result[i, j, m], source[i, j, m]))),
        )
    i, j = IndexVar("i"), IndexVar("j")
    result = TensorVar("C", fmt="sd", dtype=dtype)
    source = TensorVar("A", fmt="dd", dtype=dtype)
    if binary:
        other = TensorVar("B", fmt="dd", dtype=dtype)
        value = CINBinaryOp(Operation.ADD, source[i, j], other[i, j])
        return ForAll(i, ForAll(j, TensorAssign(result[i, j], value)))
    return ForAll(i, ForAll(j, TensorAssign(result[i, j], source[i, j])))


def build_mixed_leaf_broadcast_cin(rank3=False, dtype=torch.float32):
    """A mixed result whose compressed-parent bound comes only from C."""

    if rank3:
        i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
        result = TensorVar("C", fmt="sdd", dtype=dtype)
        source = TensorVar("A", fmt="d", dtype=dtype)
        return ForAll(
            i,
            ForAll(j, ForAll(k, TensorAssign(result[i, j, k], source[k]))),
        )
    i, j = IndexVar("i"), IndexVar("j")
    result = TensorVar("C", fmt="sd", dtype=dtype)
    source = TensorVar("A", fmt="d", dtype=dtype)
    return ForAll(i, ForAll(j, TensorAssign(result[i, j], source[j])))


def validated_mixed_storage(result, shape):
    """Assert honest compressed-parent/dense-suffix storage; return pieces."""

    rank = len(shape)
    assert str(result.index.format) == ",".join(["s"] + ["d"] * (rank - 1))
    assert tuple(result.index.mode_order) == tuple(range(rank))
    mode_indices = result.index.mode_indices
    assert len(mode_indices) == rank
    assert len(mode_indices[0]) == 2
    assert all(len(level_pair) == 0 for level_pair in mode_indices[1:])
    pos0 = mode_indices[0][0].tolist()
    crd0 = mode_indices[0][1].tolist()
    values = result.values.tolist()
    suffix = 1
    for extent in shape[1:]:
        suffix *= extent
    assert pos0 == [0, len(crd0)]
    assert crd0 == sorted(set(crd0))
    assert all(0 <= row < shape[0] for row in crd0)
    assert len(values) == len(crd0) * suffix
    return pos0, crd0, values


def mixed_oracle(kernel, dense_inputs, result_shape):
    lowering = kernel.lowering
    inputs = {
        symbol: dense.tolist()
        for symbol, dense in zip(lowering.rhs_access_symbols, dense_inputs)
    }
    outputs = run_program(
        lowering.program, inputs, {lowering.result_symbol: tuple(result_shape)}
    )
    return outputs[lowering.result_symbol]


@pytest.mark.parametrize(
    ("rank3", "shape"),
    [(False, (4, 5)), (True, (2, 3, 4))],
)
def test_mixed_leaf_broadcast_declares_every_result_loop_bound(rank3, shape):
    """Compressed parent domains cannot rely on a full-rank input bound."""

    kernel = compile_cin_via_loopir(
        build_mixed_leaf_broadcast_cin(rank3),
        shape,
        (((shape[-1],), torch.float32),),
        compile_options=auto_options(False),
    )
    for level in range(len(shape)):
        assert f"int64_t C{level}_size = result_shape[{level}];" in kernel.cpp_source
    assert "for (int64_t i = 0; i < C0_size; i++)" in kernel.cpp_source


@torch.no_grad()
@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize("shape", [(4, 5), (3, 0)])
def test_mixed_leaf_broadcast_executes_with_result_owned_parent_bound(
    regblock_enabled, shape
):
    dense = torch.randn((shape[-1],))
    result, kernel = execute_cin_via_loopir(
        build_mixed_leaf_broadcast_cin(),
        shape,
        STensor.from_torch(dense.clone(), "A").to_dense(),
        compile_options=auto_options(regblock_enabled, jit=True),
    )
    pos0, crd0, values = validated_mixed_storage(result, shape)
    expected = dense.expand(shape)
    assert torch.equal(torch.tensor(values).reshape(shape), expected)
    oracle = mixed_oracle(kernel, (dense,), shape)
    assert tuple(pos0) == oracle.positions[0]
    assert tuple(crd0) == oracle.coordinates[0]
    assert tuple(values) == oracle.values


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("rank3", [False, True])
def test_mixed_leaf_execution_matches_oracle_and_pytorch(rank3, dtype):
    torch.manual_seed(20260730)
    shape = (2, 3, 4) if rank3 else (4, 5)
    dense = torch.randn(shape, dtype=dtype)
    dense[0] = 0.0  # explicit zero-valued cells stay materialized (dense leaf)
    result, kernel = execute_cin_via_loopir(
        build_mixed_leaf_cin(rank3, dtype),
        shape,
        STensor.from_torch(dense.clone(), "A").to_dense(),
        compile_options=auto_options(False, jit=True),
    )
    pos0, crd0, values = validated_mixed_storage(result, shape)
    assert crd0 == list(range(shape[0]))
    assert torch.allclose(
        torch.tensor(values, dtype=dtype).reshape(shape),
        dense,
        atol=1e-6,
        rtol=1e-6,
    )
    oracle = mixed_oracle(kernel, (dense,), shape)
    assert tuple(pos0) == oracle.positions[0]
    assert tuple(crd0) == oracle.coordinates[0]
    assert len(values) == len(oracle.values)
    for got, expected in zip(values, oracle.values):
        assert got == pytest.approx(expected, abs=1e-6, rel=1e-6)


@torch.no_grad()
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_mixed_leaf_binary_elementwise_matches_references(regblock_enabled):
    torch.manual_seed(20260731)
    dense_a = torch.randn(4, 5)
    dense_b = torch.randn(4, 5)
    result, kernel = execute_cin_via_loopir(
        build_mixed_leaf_cin(binary=True),
        (4, 5),
        STensor.from_torch(dense_a.clone(), "A").to_dense(),
        STensor.from_torch(dense_b.clone(), "B").to_dense(),
        compile_options=auto_options(regblock_enabled, jit=True),
    )
    _, crd0, values = validated_mixed_storage(result, (4, 5))
    assert crd0 == list(range(4))
    assert torch.allclose(
        torch.tensor(values).reshape(4, 5),
        dense_a + dense_b,
        atol=1e-6,
        rtol=1e-6,
    )
    oracle = mixed_oracle(kernel, (dense_a, dense_b), (4, 5))
    assert tuple(crd0) == oracle.coordinates[0]


@torch.no_grad()
@pytest.mark.parametrize(
    ("rank3", "shape"),
    [(False, (3, 0)), (False, (0, 5)), (True, (2, 3, 0))],
)
def test_mixed_leaf_zero_extent_cells_are_canonically_empty(rank3, shape):
    dense = torch.zeros(shape)
    result, kernel = execute_cin_via_loopir(
        build_mixed_leaf_cin(rank3),
        shape,
        STensor.from_torch(dense.clone(), "A").to_dense(),
        compile_options=auto_options(False, jit=True),
    )
    pos0, crd0, values = validated_mixed_storage(result, shape)
    suffix = 1
    for extent in shape[1:]:
        suffix *= extent
    if suffix == 0 or shape[0] == 0:
        assert pos0 == [0, 0] and crd0 == [] and values == []
    oracle = mixed_oracle(kernel, (dense,), shape)
    assert tuple(pos0) == oracle.positions[0]
    assert tuple(crd0) == oracle.coordinates[0]


def test_mixed_leaf_source_is_route_stable_and_erases_to_base():
    sources = set()
    dumps = set()
    for regblock_enabled in (False, True, None):
        options = (
            CompileOptions.from_environment(environ={})
            if regblock_enabled is None
            else auto_options(regblock_enabled)
        )
        kernel = compile_cin_via_loopir(
            build_mixed_leaf_cin(),
            (4, 5),
            (((4, 5), torch.float32),),
            compile_options=options,
        )
        sources.add(kernel.cpp_source)
        if kernel.schedule is not None:
            from scorch.compiler.loopir.schedule_passes import erase_schedule

            verify_scheduled_loopir(kernel.schedule)
            assert canonical_program_dump(
                erase_schedule(kernel.schedule.program)
            ) == canonical_program_dump(kernel.schedule.base_program)
            dumps.add(canonical_program_dump(kernel.schedule.program))
    assert len(sources) == 1
    assert len(dumps) <= 1
    replay = compile_cin_via_loopir(
        build_mixed_leaf_cin(),
        (4, 5),
        (((4, 5), torch.float32),),
        compile_options=auto_options(False),
    )
    assert replay.cpp_source in sources


@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_mixed_leaf_adjacent_seams_stay_fail_closed(regblock_enabled):
    from scorch.compiler.loopir.lower_cin import LoopIRLoweringError

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    reduction = ForAll(
        k,
        ForAll(
            i,
            ForAll(
                j,
                TensorAssign(
                    TensorVar("C", fmt="sd")[k, j],
                    TensorVar("A", fmt="ddd")[k, i, j],
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(
            reduction,
            (6, 5),
            (((6, 4, 5), torch.float32),),
            compile_options=auto_options(regblock_enabled),
        )
    assert error.value.defect.code == "unsupported_sparse_output_reduction"

    i2, j2 = IndexVar("i"), IndexVar("j")
    sd_operand = ForAll(
        i2,
        ForAll(
            j2,
            TensorAssign(
                TensorVar("C", fmt="dd")[i2, j2],
                TensorVar("A", fmt="sd")[i2, j2],
            ),
        ),
    )
    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(
            sd_operand,
            (4, 5),
            (((4, 5), torch.float32),),
            compile_options=auto_options(regblock_enabled),
        )
    assert error.value.defect.code == "unsupported_format"

    i3, j3 = IndexVar("i"), IndexVar("j")
    sparse_domain = ForAll(
        i3,
        ForAll(
            j3,
            TensorAssign(
                TensorVar("C", fmt="sd")[i3, j3],
                TensorVar("A", fmt="ss")[i3, j3],
            ),
        ),
    )
    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(
            sparse_domain,
            (4, 5),
            (((4, 5), torch.float32),),
            compile_options=auto_options(regblock_enabled),
        )
    assert error.value.defect.code == "unsupported_sparse_output_domain"


def test_mixed_leaf_legacy_comparand_is_failure_evidence_only():
    """The defective legacy source is retained evidence, never an oracle.

    The legacy pipeline still generates a kernel for this family whose
    assembly is inconsistent: every dense-leaf value is appended, but the
    compressed parent's coordinates are never assembled, so the returned
    storage would carry values with no owning rows.  This locks the
    defect's shape as evidence and the intentional absence of any byte
    or execution parity gate for the B2 family.
    """

    legacy_cpp = legacy_generated_cpp(
        build_mixed_leaf_cin(),
        (4, 5),
        (((4, 5), torch.float32),),
        compile_options=auto_options(False),
    )
    assert "C_values.emplace_back" in legacy_cpp
    assert "C0_crd.push_back" not in legacy_cpp
    assert "C0_crd.emplace_back" not in legacy_cpp

    loopir_kernel = compile_cin_via_loopir(
        build_mixed_leaf_cin(),
        (4, 5),
        (((4, 5), torch.float32),),
        compile_options=auto_options(False),
    )
    assert "C0_crd.push_back(i);" in loopir_kernel.cpp_source
    assert loopir_kernel.cpp_source != legacy_cpp


def build_mixed_leaf_program(forge=None):
    """Hand-build the exact mixed dense-leaf chain for target adversaries."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    symbol_a = builder.new_symbol_id()
    symbol_c = builder.new_symbol_id()
    decl_a = builder.tensor(
        symbol_a,
        "A",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (builder.level(LevelKind.DENSE, 0), builder.level(LevelKind.DENSE, 1)),
    )
    decl_c = builder.tensor(
        symbol_c,
        "C",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.COMPRESSED, 0),
            builder.level(LevelKind.DENSE, 1),
        ),
    )
    index_i = builder.new_index_id()
    index_j = builder.new_index_id()
    coords = (builder.index_value(index_i), builder.index_value(index_j))
    if forge == "swapped_coords":
        coords = (coords[1], coords[0])
    append = builder.append_entry(
        symbol_c,
        coords,
        builder.load(
            symbol_a,
            (builder.index_value(index_i), builder.index_value(index_j)),
        ),
    )
    inner_statements = (append,)
    inner = builder.dense_for(index_j, dim_j.dimension, builder.block(inner_statements))
    outer = builder.dense_for(index_i, dim_i.dimension, builder.block((inner,)))
    return (
        builder.program(
            (dim_i, dim_j),
            (decl_a, decl_c),
            (symbol_a,),
            (symbol_c,),
            builder.block((outer,)),
        ),
        symbol_a,
    )


def build_mixed_leaf_colliding_dimension_program():
    """Two distinct binders share one display dimension and would shadow."""

    builder = LoopIRBuilder()
    dimension = builder.dimension("q")
    symbol_a = builder.new_symbol_id()
    symbol_c = builder.new_symbol_id()
    decl_a = builder.tensor(
        symbol_a,
        "A",
        ScalarType.FLOAT32,
        (dimension.dimension,),
        (builder.level(LevelKind.DENSE, 0),),
    )
    decl_c = builder.tensor(
        symbol_c,
        "C",
        ScalarType.FLOAT32,
        (dimension.dimension, dimension.dimension),
        (
            builder.level(LevelKind.COMPRESSED, 0),
            builder.level(LevelKind.DENSE, 1),
        ),
    )
    outer_index = builder.new_index_id()
    inner_index = builder.new_index_id()
    append = builder.append_entry(
        symbol_c,
        (
            builder.index_value(outer_index),
            builder.index_value(inner_index),
        ),
        # At the leaf, an unguarded target would spell this outer-coordinate
        # load with the inner loop's shadowing ``q`` variable.
        builder.load(symbol_a, (builder.index_value(outer_index),)),
    )
    inner = builder.dense_for(
        inner_index,
        dimension.dimension,
        builder.block((append,)),
    )
    outer = builder.dense_for(
        outer_index,
        dimension.dimension,
        builder.block((inner,)),
    )
    return (
        builder.program(
            (dimension,),
            (decl_a, decl_c),
            (symbol_a,),
            (symbol_c,),
            builder.block((outer,)),
        ),
        symbol_a,
    )


def test_mixed_leaf_rejects_distinct_binders_with_one_cpp_name():
    program, symbol_a = build_mixed_leaf_colliding_dimension_program()
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes={symbol_a: (3,)},
            result_shape=(3, 3),
            compile_options=CompileOptions.from_environment(environ={}),
        )
    assert error.value.defect.code == "generated_name_collision"


def test_mixed_leaf_hand_built_program_matches_pipeline_source():
    program, symbol_a = build_mixed_leaf_program()
    function = lower_loopir_to_llir(
        program,
        input_shapes={symbol_a: (4, 5)},
        result_shape=(4, 5),
        compile_options=CompileOptions.from_environment(environ={}),
    )
    from scorch.ops import _lower_generated_llir

    options = CompileOptions.from_environment(environ={})
    cpp = _lower_generated_llir(function, options, CompilationContext(options))
    kernel = compile_cin_via_loopir(
        build_mixed_leaf_cin(),
        (4, 5),
        (((4, 5), torch.float32),),
        compile_options=auto_options(False),
    )
    assert cpp == kernel.cpp_source


def test_mixed_leaf_swapped_append_coordinates_fail_closed():
    program, symbol_a = build_mixed_leaf_program(forge="swapped_coords")
    with pytest.raises(Exception) as error:
        lower_loopir_to_llir(
            program,
            input_shapes={symbol_a: (4, 5)},
            result_shape=(4, 5),
            compile_options=CompileOptions.from_environment(environ={}),
        )
    code = getattr(getattr(error.value, "defect", None), "code", None)
    # The verifier owns this boundary: a swapped append coordinate is a
    # coordinate-domain violation before target routing can even run.
    assert code in ("unsupported_program_shape", "domain_mismatch")
