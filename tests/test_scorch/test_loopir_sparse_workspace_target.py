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


def build_spmspm_cin(dtype=torch.float32):
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    a = TensorVar("A", fmt="ss", dtype=dtype)
    b = TensorVar("B", fmt="ss", dtype=dtype)
    c = TensorVar("C", fmt="ss", dtype=dtype)
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=Operation.ADD
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


def execute_legacy_module(kernel, result_shape, st_a, st_b, options):
    """Safe legacy execution: run the byte-identical legacy kernel raw."""

    from scorch.ops import (
        _load_validated_prepared_kernel,
        _prepare_generated_kernel_build,
        _snapshot_runtime_tensors,
    )

    context = CompilationContext(options)
    prepared = _prepare_generated_kernel_build(
        options.build.preamble_source,
        kernel.cpp_source,
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

    # Safe legacy execution: the byte-identical legacy kernel, run raw.
    legacy_raw = execute_legacy_module(
        kernel,
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
