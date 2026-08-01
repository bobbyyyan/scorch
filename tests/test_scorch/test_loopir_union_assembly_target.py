"""Phase-7 rank-2+ UNION assembly: ordered one-sided tails and descent.

Elementwise ADD union chains into dense-prefix/multi-compressed-suffix
results (``ss``, ``sss``, ``dss``, ``ssss``, ``dsss``): every compressed
result level unites the same two stored operand streams through a
two-cursor UNION merge binding both cursors' positions.  An aligned
cursor anchors its child descent; an unaligned cursor contributes the
empty child stream, so one-sided cases and post-exhaustion tails drain
the surviving operand's whole subtree in stored order behind the same
conditional parent append every assembly level owns.

The legacy comparand is honest for this family (its generated union
kernels match an independent ordered-union reference across overlapping,
disjoint, one-sided, and empty supports — the pre-implementation census),
so the gate is byte parity with ``legacy_generated_cpp`` in both
automatic policy arms plus the production LoopIR oracle and the PyTorch
dense reference.  No public operation spells elementwise sparse ADD
today, so the compiled differentials run through the shared production
JIT path against PyTorch; the compiler-level CIN entry is the caller.
The family is serial: the legacy route emits no OpenMP marking for it.
"""

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
from scorch.compiler.compilation_context import CompilationContext
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import (
    LoopIRTargetError,
    lower_loopir_to_llir,
)
from scorch.compiler.loopir.nodes import LevelKind, MergeMode, ScalarType
from scorch.compiler.loopir.nodes import BinaryOp as LoopIRBinaryOp
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compare_generated_sources,
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.schedule_passes import erase_schedule
from scorch.compiler.loopir.verifier import LoopIRVerificationError
from scorch.stensor import STensor
from scorch.storage import TensorIndex
from tests.test_scorch.test_loopir_multi_compressed_target import (
    validated_storage_pieces,
)
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

_KIND = {"d": LevelKind.DENSE, "s": LevelKind.COMPRESSED}

_SHAPES = {
    "ss": (4, 5),
    "sss": (3, 4, 5),
    "dss": (3, 4, 5),
    "ssss": (2, 3, 4, 5),
    "dsss": (2, 3, 4, 5),
}


def build_union_cin(fmt, dtype=torch.float32, *, commuted=False):
    """C = A + B elementwise over one shared format (the union shape)."""

    rank = len(fmt)
    ivars = tuple(IndexVar(name) for name in "ijkl"[:rank])
    a = TensorVar("A", fmt=fmt, dtype=dtype)
    b = TensorVar("B", fmt=fmt, dtype=dtype)
    c = TensorVar("C", fmt=fmt, dtype=dtype)
    left, right = (b, a) if commuted else (a, b)
    stmt = TensorAssign(c[ivars], CINBinaryOp(Operation.ADD, left[ivars], right[ivars]))
    for index_var in reversed(ivars):
        stmt = ForAll(index_var, stmt)
    return stmt


def sparse(dense, name, fmt):
    return STensor.from_torch(dense.clone(), name).to_sparse(fmt)


def executed(cin, result_shape, stensors, regblock_enabled):
    out = execute_cin_via_loopir(
        cin,
        result_shape,
        *stensors,
        compile_options=auto_options(regblock_enabled, jit=True),
    )
    return out[0] if isinstance(out, tuple) else out


def oracle_storage(kernel, dense_inputs, fmt, result_shape):
    """Base and scheduled oracle agreement; returns the level storage."""

    lowering = kernel.lowering
    kinds = tuple(_KIND[ch] for ch in fmt)
    inputs = {}
    for symbol, dense in zip(lowering.rhs_access_symbols, dense_inputs):
        inputs[symbol] = LevelTensorStorage.from_dense(
            dense.tolist(), tuple(dense.shape), tuple(range(len(fmt))), kinds
        )
    base = run_program(
        lowering.program, inputs, {lowering.result_symbol: tuple(result_shape)}
    )
    result = base[lowering.result_symbol]
    if kernel.schedule is not None:
        scheduled = run_program(
            kernel.schedule.program,
            inputs,
            {lowering.result_symbol: tuple(result_shape)},
        )
        assert scheduled[lowering.result_symbol] == result
    return result


def _fixtures(fmt, seed=20260731, dtype=torch.float32):
    torch.manual_seed(seed)
    shape = _SHAPES[fmt]
    dense_a = ((torch.rand(shape) < 0.4) * torch.randn(shape)).to(dtype)
    dense_b = ((torch.rand(shape) < 0.4) * torch.randn(shape)).to(dtype)
    return dense_a, dense_b


def _disjoint_fixtures(fmt, seed=77):
    torch.manual_seed(seed)
    shape = _SHAPES[fmt]
    numel = 1
    for extent in shape:
        numel *= extent
    parity = (torch.arange(numel) % 2 == 0).reshape(shape)
    return torch.randn(shape) * parity, torch.randn(shape) * (~parity)


def _one_sided_row_fixtures(fmt, seed=78):
    torch.manual_seed(seed)
    shape = _SHAPES[fmt]
    dense_a = torch.zeros(shape)
    dense_b = torch.zeros(shape)
    for row in range(shape[0]):
        if row % 2 == 0:
            dense_a[row] = torch.randn(shape[1:])
        else:
            dense_b[row] = torch.randn(shape[1:])
    return dense_a, dense_b


def _column_tail_fixtures(fmt, seed=79):
    torch.manual_seed(seed)
    shape = _SHAPES[fmt]
    half = shape[-1] // 2
    dense_a = torch.zeros(shape)
    dense_b = torch.zeros(shape)
    dense_a[..., :half] = torch.randn(shape[:-1] + (half,))
    dense_b[..., half:] = torch.randn(shape[:-1] + (shape[-1] - half,))
    return dense_a, dense_b


# -- source parity -----------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fmt", ["ss", "sss", "dss", "ssss", "dsss"])
def test_source_parity_matches_legacy_in_both_arms(fmt, dtype):
    shape = _SHAPES[fmt]
    bindings = ((shape, dtype), (shape, dtype))
    for arm in (False, True):
        comparison = compare_generated_sources(
            build_union_cin(fmt, dtype),
            shape,
            bindings,
            compile_options=auto_options(arm),
        )
        assert comparison.identical, f"fmt {fmt} arm regblock={arm} diverged"


@pytest.mark.parametrize("fmt", ["ss", "dss"])
def test_commuted_operands_hold_parity(fmt):
    shape = _SHAPES[fmt]
    bindings = ((shape, torch.float32), (shape, torch.float32))
    for arm in (False, True):
        comparison = compare_generated_sources(
            build_union_cin(fmt, commuted=True),
            shape,
            bindings,
            compile_options=auto_options(arm),
        )
        assert comparison.identical


def test_family_is_serial_and_route_stable():
    """The union family emits no OpenMP marking, in either arm."""

    sources = set()
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            build_union_cin("ss"),
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        assert "#pragma omp" not in kernel.cpp_source
        assert "C0_crd.push_back" in kernel.cpp_source
        # The three-case union body plus one post-exhaustion tail per side.
        assert kernel.cpp_source.count("while (pA0 < pA0_end && pB0 < pB0_end)") == 1
        assert kernel.cpp_source.count("while (pA0 < pA0_end)") == 1
        assert kernel.cpp_source.count("while (pB0 < pB0_end)") == 1
        sources.add(kernel.cpp_source)
        replay = compile_cin_via_loopir(
            build_union_cin("ss"),
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        assert replay.cpp_source in sources


def test_canonical_dump_is_arm_stable_and_erases_to_base():
    dumps = []
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            build_union_cin("sss"),
            (3, 4, 5),
            (((3, 4, 5), torch.float32), ((3, 4, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        dumps.append(kernel.program_dump)
        program = (
            kernel.schedule.program
            if kernel.schedule is not None
            else kernel.lowering.program
        )
        erased = erase_schedule(program)
        assert canonical_program_dump(erased) == canonical_program_dump(
            kernel.lowering.program
        )
    assert dumps[0] == dumps[1]
    assert '"mode":"union"' in dumps[0]
    assert '"positions"' in dumps[0]
    assert '"kind":"append_entry"' in dumps[0]


def test_compile_is_deterministic_per_arm():
    for arm in (False, True):
        first = compile_cin_via_loopir(
            build_union_cin("ss"),
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        second = compile_cin_via_loopir(
            build_union_cin("ss"),
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        assert first.cpp_source == second.cpp_source
        assert first.program_dump == second.program_dump


# -- compiled execution ------------------------------------------------------


@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize("fmt", ["ss", "sss", "dss"])
def test_compiled_execution_matches_every_reference(regblock_enabled, fmt):
    dense_a, dense_b = _fixtures(fmt)
    shape = _SHAPES[fmt]
    cin = build_union_cin(fmt)
    result = executed(
        cin,
        shape,
        (sparse(dense_a, "A", fmt), sparse(dense_b, "B", fmt)),
        regblock_enabled,
    )
    pieces, values = validated_storage_pieces(result, fmt, shape)
    assert torch.allclose(result.to_torch(), dense_a + dense_b, atol=1e-3, rtol=1e-3)

    kernel = compile_cin_via_loopir(
        cin,
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(regblock_enabled),
    )
    oracle = oracle_storage(kernel, (dense_a, dense_b), fmt, shape)
    for level, ch in enumerate(fmt):
        if ch != "s":
            continue
        pos, crd = pieces[level]
        assert tuple(pos) == oracle.positions[level]
        assert tuple(crd) == oracle.coordinates[level]
    assert len(values) == len(oracle.values)
    for got, expected in zip(values, oracle.values):
        assert got == pytest.approx(expected, abs=1e-5, rel=1e-5)


@pytest.mark.parametrize(
    "fixture_builder",
    [_disjoint_fixtures, _one_sided_row_fixtures, _column_tail_fixtures],
    ids=["disjoint", "one-sided-rows", "column-tails"],
)
@pytest.mark.parametrize("fmt", ["ss", "sss", "dss", "ssss"])
def test_one_sided_supports_execute_exactly(fmt, fixture_builder):
    """Disjoint and one-sided supports drain deterministic ordered tails."""

    dense_a, dense_b = fixture_builder(fmt)
    shape = _SHAPES[fmt]
    result = executed(
        build_union_cin(fmt),
        shape,
        (sparse(dense_a, "A", fmt), sparse(dense_b, "B", fmt)),
        False,
    )
    pieces, values = validated_storage_pieces(result, fmt, shape)
    assert torch.allclose(result.to_torch(), dense_a + dense_b, atol=1e-3, rtol=1e-3)

    kernel = compile_cin_via_loopir(
        build_union_cin(fmt),
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(False),
    )
    oracle = oracle_storage(kernel, (dense_a, dense_b), fmt, shape)
    for level, ch in enumerate(fmt):
        if ch != "s":
            continue
        pos, crd = pieces[level]
        assert tuple(pos) == oracle.positions[level]
        assert tuple(crd) == oracle.coordinates[level]
    assert len(values) == len(oracle.values)


@pytest.mark.parametrize("empty_side", ["a", "b"])
def test_one_sided_exhaustion_from_the_start(empty_side):
    """An entirely empty operand leaves the other stream drained in order."""

    dense_a, dense_b = _fixtures("sss", seed=13)
    shape = _SHAPES["sss"]
    if empty_side == "a":
        dense_a = torch.zeros(shape)
    else:
        dense_b = torch.zeros(shape)
    result = executed(
        build_union_cin("sss"),
        shape,
        (sparse(dense_a, "A", "sss"), sparse(dense_b, "B", "sss")),
        False,
    )
    validated_storage_pieces(result, "sss", shape)
    assert torch.allclose(result.to_torch(), dense_a + dense_b, atol=1e-3, rtol=1e-3)


def test_rank4_union_executes():
    dense_a, dense_b = _fixtures("ssss")
    result = executed(
        build_union_cin("ssss"),
        _SHAPES["ssss"],
        (sparse(dense_a, "A", "ssss"), sparse(dense_b, "B", "ssss")),
        False,
    )
    validated_storage_pieces(result, "ssss", _SHAPES["ssss"])
    assert torch.allclose(result.to_torch(), dense_a + dense_b, atol=1e-3, rtol=1e-3)


def test_float64_execution_matches_reference():
    dense_a, dense_b = _fixtures("ss", dtype=torch.float64)
    result = executed(
        build_union_cin("ss", torch.float64),
        (4, 5),
        (sparse(dense_a, "A", "ss"), sparse(dense_b, "B", "ss")),
        False,
    )
    assert torch.allclose(result.to_torch(), dense_a + dense_b, atol=1e-9, rtol=1e-9)


def test_cancellation_retains_the_explicit_zero():
    """Aligned entries summing to zero stay stored, exactly like legacy."""

    dense_a = torch.zeros(3, 4)
    dense_b = torch.zeros(3, 4)
    dense_a[1, 2] = 3.0
    dense_b[1, 2] = -3.0
    dense_a[2, 0] = 1.0
    result = executed(
        build_union_cin("ss"),
        (3, 4),
        (sparse(dense_a, "A", "ss"), sparse(dense_b, "B", "ss")),
        False,
    )
    assert result.storage.index.mode_indices[0][1].tolist() == [1, 2]
    assert result.storage.index.mode_indices[1][1].tolist() == [2, 0]
    assert result.storage.value.tolist() == [0.0, 1.0]


def test_explicit_zero_operand_entries_survive_the_union():
    """A hand-built stored zero unions into a stored zero entry."""

    pos = torch.tensor([0, 1], dtype=torch.int32)
    crd = torch.tensor([1], dtype=torch.int32)
    pos1 = torch.tensor([0, 1], dtype=torch.int32)
    crd1 = torch.tensor([2], dtype=torch.int32)
    explicit_zero = STensor(
        name="A",
        shape=(3, 4),
        index=TensorIndex("ss", [[pos, crd], [pos1, crd1]]),
        value=torch.tensor([0.0]),
    )
    dense_b = torch.zeros(3, 4)
    dense_b[2, 3] = 4.0
    result = executed(
        build_union_cin("ss"),
        (3, 4),
        (explicit_zero, sparse(dense_b, "B", "ss")),
        False,
    )
    assert result.storage.index.mode_indices[0][1].tolist() == [1, 2]
    assert result.storage.index.mode_indices[1][1].tolist() == [2, 3]
    assert result.storage.value.tolist() == [0.0, 4.0]


def test_empty_intermediate_parents_are_suppressed():
    """A hand-built parent with an empty child segment appends nothing."""

    pos0 = torch.tensor([0, 2], dtype=torch.int32)
    crd0 = torch.tensor([0, 2], dtype=torch.int32)
    pos1 = torch.tensor([0, 0, 1], dtype=torch.int32)  # row 0 stored empty
    crd1 = torch.tensor([3], dtype=torch.int32)
    operand_a = STensor(
        name="A",
        shape=(3, 4),
        index=TensorIndex("ss", [[pos0, crd0], [pos1, crd1]]),
        value=torch.tensor([7.0]),
    )
    dense_b = torch.zeros(3, 4)
    dense_b[1, 1] = 2.0
    result = executed(
        build_union_cin("ss"),
        (3, 4),
        (operand_a, sparse(dense_b, "B", "ss")),
        False,
    )
    assert result.storage.index.mode_indices[0][1].tolist() == [1, 2]
    assert result.storage.index.mode_indices[1][0].tolist() == [0, 1, 2]
    assert result.storage.index.mode_indices[1][1].tolist() == [1, 3]
    assert result.storage.value.tolist() == [2.0, 7.0]


def test_dense_prefix_empty_rows_keep_full_position_arrays():
    """The dss union closes every dense row's position slot."""

    shape = (3, 4, 5)
    dense_a = torch.zeros(shape)
    dense_b = torch.zeros(shape)
    dense_a[2, 1, 3] = 4.0
    dense_b[2, 1, 0] = 5.0
    result = executed(
        build_union_cin("dss"),
        shape,
        (sparse(dense_a, "A", "dss"), sparse(dense_b, "B", "dss")),
        False,
    )
    mode_indices = result.storage.index.mode_indices
    assert mode_indices[1][0].tolist() == [0, 0, 0, 1]
    assert mode_indices[1][1].tolist() == [1]
    assert mode_indices[2][0].tolist() == [0, 2]
    assert mode_indices[2][1].tolist() == [0, 3]
    assert result.storage.value.tolist() == [5.0, 4.0]


def test_zero_extent_cells_execute():
    dense = torch.zeros(0, 3)
    result = executed(
        build_union_cin("ss"),
        (0, 3),
        (sparse(dense, "A", "ss"), sparse(dense, "B", "ss")),
        False,
    )
    assert result.storage.index.mode_indices[0][0].tolist() == [0, 0]
    assert result.storage.value.tolist() == []


def test_deterministic_storage_and_repeated_compiled_differential():
    """Repeated compiled runs are byte-stable and match the dense reference."""

    dense_a, dense_b = _fixtures("sss", seed=7)
    reference = dense_a + dense_b
    snapshots = []
    for _ in range(3):
        result = executed(
            build_union_cin("sss"),
            _SHAPES["sss"],
            (sparse(dense_a, "A", "sss"), sparse(dense_b, "B", "sss")),
            False,
        )
        assert torch.allclose(result.to_torch(), reference, atol=1e-3, rtol=1e-3)
        snapshots.append(
            (
                [
                    [tensor.tolist() for tensor in level]
                    for level in result.storage.index.mode_indices
                ],
                result.storage.value.tolist(),
            )
        )
    assert snapshots[0] == snapshots[1] == snapshots[2]


# -- target-owned checked output mutations -----------------------------------

_CHECKED_MUTATION_CENSUS = {
    # fmt -> (leaf level, leaf append count, parent pushes, vector sets)
    "ss": (1, 9, (("C0_crd.push_back", 5),), 8),
    "dss": (2, 9, (("C1_crd.push_back", 5),), 9),
    "ssss": (
        3,
        17,
        (
            ("C0_crd.push_back", 5),
            ("C1_crd.push_back", 9),
            ("C2_crd.push_back", 13),
        ),
        32,
    ),
}


@pytest.mark.parametrize("fmt", ["ss", "dss", "ssss"])
def test_union_enters_dynamic_pass_with_only_checked_mutations(fmt, monkeypatch):
    """Every union case and tail is safe before the generic rewrite."""

    import scorch.compiler.llir_pass_manager as pass_manager

    from tests.test_scorch.test_loopir_multi_compressed_target import (
        _b3_output_mutation_census,
    )

    original = pass_manager.rewrite_dynamic_vector_accesses
    censuses = []

    def capture(value, context):
        censuses.append(_b3_output_mutation_census(value))
        return original(value, context)

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        capture,
    )
    shape = _SHAPES[fmt]
    compile_cin_via_loopir(
        build_union_cin(fmt),
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(False),
    )
    assert len(censuses) == 1
    unchecked, calls = censuses[0]
    assert unchecked == []
    leaf_level, leaf_appends, parent_pushes, vector_sets = _CHECKED_MUTATION_CENSUS[fmt]
    assert calls.count("C_values.emplace_back") == leaf_appends
    assert calls.count(f"C{leaf_level}_crd.emplace_back") == leaf_appends
    for parent_push, count in parent_pushes:
        assert calls.count(parent_push) == count
    assert calls.count("scorch_vector_set") == vector_sets


def test_union_dynamic_pass_is_byte_neutral(monkeypatch):
    """Omitting the generic rewrite leaves byte-exact safe output."""

    import scorch.compiler.llir_pass_manager as pass_manager

    baseline = compile_cin_via_loopir(
        build_union_cin("dss"),
        (3, 4, 5),
        (((3, 4, 5), torch.float32), ((3, 4, 5), torch.float32)),
        compile_options=auto_options(False),
    )

    def omit(value, context):
        return value

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        omit,
    )
    omitted = compile_cin_via_loopir(
        build_union_cin("dss"),
        (3, 4, 5),
        (((3, 4, 5), torch.float32), ((3, 4, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    assert omitted.cpp_source == baseline.cpp_source


# -- hand-built target programs ----------------------------------------------


def build_union_assembly_program(*, forge_positions=None, forge_default=None):
    """The exact ss+ss->ss union LoopIR program the pipeline lowers."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    cc_levels = lambda: (  # noqa: E731
        builder.level(LevelKind.COMPRESSED, 0),
        builder.level(LevelKind.COMPRESSED, 1),
    )
    decl_a = builder.tensor(
        a, "A", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), cc_levels()
    )
    decl_b = builder.tensor(
        b, "B", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), cc_levels()
    )
    decl_c = builder.tensor(
        c, "C", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), cc_levels()
    )
    index_i = builder.new_index_id()
    index_j = builder.new_index_id()
    position_a0 = builder.new_position_id()
    position_b0 = builder.new_position_id()
    cursor_a0 = builder.sparse_cursor(
        builder.new_cursor_id(), a, 0, builder.root_position()
    )
    cursor_b0 = builder.sparse_cursor(
        builder.new_cursor_id(), b, 0, builder.root_position()
    )
    cursor_a1 = builder.sparse_cursor(
        builder.new_cursor_id(), a, 1, builder.position_value(position_a0)
    )
    cursor_b1 = builder.sparse_cursor(
        builder.new_cursor_id(), b, 1, builder.position_value(position_b0)
    )
    default = forge_default if forge_default is not None else 0.0
    leaf = builder.append_entry(
        c,
        (builder.index_value(index_i), builder.index_value(index_j)),
        builder.binary(
            LoopIRBinaryOp.ADD,
            builder.cursor_value(cursor_a1.cursor, builder.float_const(default)),
            builder.cursor_value(cursor_b1.cursor, builder.float_const(default)),
        ),
    )
    inner_positions = (
        forge_positions
        if forge_positions is not None
        else (builder.new_position_id(), builder.new_position_id())
    )
    inner = builder.merged_sparse_for(
        MergeMode.UNION,
        (cursor_a1, cursor_b1),
        index_j,
        builder.block((leaf,)),
        inner_positions,
    )
    outer = builder.merged_sparse_for(
        MergeMode.UNION,
        (cursor_a0, cursor_b0),
        index_i,
        builder.block((inner,)),
        (position_a0, position_b0),
    )
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_b, decl_c),
        (a, b),
        (c,),
        builder.block((outer,)),
    )
    return program, builder, (a, b, c)


def test_hand_built_program_lowers_and_matches_pipeline_source():
    program, _, (a, b, _) = build_union_assembly_program()
    options = CompileOptions.from_environment(environ={})
    context = CompilationContext(options)
    function = lower_loopir_to_llir(
        program,
        input_shapes={a: (4, 5), b: (4, 5)},
        result_shape=(4, 5),
        compile_options=options,
        compilation_context=context,
    )
    from scorch.ops import _lower_generated_llir

    cpp_source = _lower_generated_llir(function, options, context)
    kernel = compile_cin_via_loopir(
        build_union_cin("ss"),
        (4, 5),
        (((4, 5), torch.float32), ((4, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    assert cpp_source == kernel.cpp_source


def test_hand_built_program_executes_one_sided_descent_on_the_oracle():
    program, _, (a, b, c) = build_union_assembly_program()
    storage_a = LevelTensorStorage.from_dense(
        [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        (3, 3),
        (0, 1),
        (LevelKind.COMPRESSED, LevelKind.COMPRESSED),
    )
    storage_b = LevelTensorStorage.from_dense(
        [[0.0, 4.0, 2.0], [5.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        (3, 3),
        (0, 1),
        (LevelKind.COMPRESSED, LevelKind.COMPRESSED),
    )
    results = run_program(program, {a: storage_a, b: storage_b}, {c: (3, 3)})
    result = results[c]
    assert result.positions[0] == (0, 3)
    assert result.coordinates[0] == (0, 1, 2)
    assert result.positions[1] == (0, 3, 4, 5)
    assert result.coordinates[1] == (0, 1, 2, 0, 1)
    assert result.values == (1.0, 4.0, 4.0, 5.0, 3.0)


def test_one_bound_one_none_union_positions_fail_closed():
    forged, _, (a2, b2, c2) = build_union_assembly_program()
    inner = forged.body.statements[0].body.statements[0]
    object.__setattr__(inner, "positions", (inner.positions[0], None))
    with pytest.raises(LoopIRVerificationError) as error:
        run_program(
            forged,
            {a2: object(), b2: object()},
            {c2: (4, 5)},
        )
    assert error.value.defect.code == "unsupported_sparse_hierarchy"


def test_nonzero_union_default_fails_closed_at_the_target():
    program, _, (a, b, _) = build_union_assembly_program(forge_default=1.0)
    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes={a: (4, 5), b: (4, 5)},
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=CompilationContext(options),
        )
    assert error.value.defect.code == "unsupported_union_default"


def test_mixed_union_and_intersection_chain_fails_closed():
    """An intersected level above a united level has no one-sided iterator.

    (Forging the inner level instead dies even earlier: the verifier
    rejects the now-dead union defaults on intersected leaf cursors.)
    """

    program, builder, (a, b, _) = build_union_assembly_program()
    outer = program.body.statements[0]
    object.__setattr__(outer, "mode", MergeMode.INTERSECTION)
    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes={a: (4, 5), b: (4, 5)},
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=CompilationContext(options),
        )
    assert error.value.defect.code == "unsupported_program_shape"


# -- adjacent seams stay fail-closed -----------------------------------------


def _seam_cells():
    i, j = IndexVar("i"), IndexVar("j")
    f32 = torch.float32
    subtraction = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("C", fmt="ss")[i, j],
                CINBinaryOp(
                    Operation.SUB,
                    TensorVar("A", fmt="ss")[i, j],
                    TensorVar("B", fmt="ss")[i, j],
                ),
            ),
        ),
    )
    union_with_dense = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("C", fmt="ss")[i, j],
                CINBinaryOp(
                    Operation.ADD,
                    TensorVar("A", fmt="ss")[i, j],
                    TensorVar("B", fmt="dd")[i, j],
                ),
            ),
        ),
    )
    three = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("C", fmt="ss")[i, j],
                CINBinaryOp(
                    Operation.ADD,
                    CINBinaryOp(
                        Operation.ADD,
                        TensorVar("A", fmt="ss")[i, j],
                        TensorVar("B", fmt="ss")[i, j],
                    ),
                    TensorVar("D", fmt="ss")[i, j],
                ),
            ),
        ),
    )
    i1 = IndexVar("i")
    rank1 = ForAll(
        i1,
        TensorAssign(
            TensorVar("C", fmt="s")[i1],
            CINBinaryOp(
                Operation.ADD,
                TensorVar("A", fmt="s")[i1],
                TensorVar("B", fmt="s")[i1],
            ),
        ),
    )
    mixed_suffix = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("C", fmt="ss")[i, j],
                CINBinaryOp(
                    Operation.MUL,
                    CINBinaryOp(
                        Operation.ADD,
                        TensorVar("A", fmt="ss")[i, j],
                        TensorVar("B", fmt="ss")[i, j],
                    ),
                    TensorVar("D", fmt="dd")[i, j],
                ),
            ),
        ),
    )
    return [
        (
            "sparse_subtraction",
            subtraction,
            (4, 5),
            (((4, 5), f32), ((4, 5), f32)),
            "unsupported_sparse_subtraction",
        ),
        (
            "union_with_dense",
            union_with_dense,
            (4, 5),
            (((4, 5), f32), ((4, 5), f32)),
            "unsupported_union_with_dense",
        ),
        # The three-operand union previously died at the layout seam
        # (unsupported_sparse_output); the united classification now admits
        # its domains, and the target's two-cursor boundary owns the
        # rejection, exactly like the three-operand intersection.
        (
            "three_operand_union",
            three,
            (4, 5),
            (((4, 5), f32), ((4, 5), f32), ((4, 5), f32)),
            "unsupported_program_shape",
        ),
        (
            "rank1_union_output",
            rank1,
            (4,),
            (((4,), f32), ((4,), f32)),
            "unsupported_sparse_output",
        ),
        # (A + B) * D over a dense factor keeps 2-cursor united domains
        # (the dense operand defers in the lattice), but the united leaf
        # envelope is exactly the two-operand sum; the wider expression
        # stays fail-closed at the target's leaf-shape boundary.
        (
            "union_times_dense_factor",
            mixed_suffix,
            (4, 5),
            (((4, 5), f32), ((4, 5), f32), ((4, 5), f32)),
            "unsupported_program_shape",
        ),
    ]


@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize("cell", _seam_cells(), ids=[cell[0] for cell in _seam_cells()])
def test_adjacent_seams_stay_fail_closed(regblock_enabled, cell):
    _, cin, result_shape, bindings, expected = cell
    with pytest.raises((LoopIRLoweringError, LoopIRTargetError)) as error:
        compile_cin_via_loopir(
            cin,
            result_shape,
            bindings,
            compile_options=auto_options(regblock_enabled),
        )
    assert error.value.defect.code == expected
