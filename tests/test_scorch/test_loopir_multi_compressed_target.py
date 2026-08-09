"""Phase-7 multi-compressed intersection assembly: the B3 family.

Elementwise MUL intersection chains into dense-prefix/multi-compressed-
suffix results (``ss``, ``sss``, ``dss``, ``ssss``) — the output formats
production ``einsum`` infers for same-shape MUL over compressed inputs —
lower to nested two-cursor INTERSECTION merges over an ordered
:class:`AppendEntry` leaf.  Each merged level binds both aligned cursor
positions so child levels descend from them; one conditional compressed-
parent append per structural level materializes a parent coordinate
exactly when its child intersection appended entries, so empty child
intersections cascade upward and never fabricate parents.

The legacy comparand is honest for this family, so the gate is byte
parity with ``legacy_generated_cpp`` in both automatic policy arms plus
the production LoopIR oracle (exact multi-level storage) and the PyTorch
dense reference — the B1 discipline, not the B2 no-parity disposition.
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
from scorch.compiler.compilation_context import (
    CompilationContext,
    CompilerStageId,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import (
    LoopIRTargetError,
    lower_loopir_to_llir,
)
from scorch.compiler.loopir.nodes import (
    LevelKind,
    MergeMode,
    ScalarType,
)
from scorch.compiler.loopir.nodes import BinaryOp as LoopIRBinaryOp
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compare_generated_sources,
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.schedule_passes import erase_schedule
from scorch.stensor import STensor
from scorch.storage import TensorIndex
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

_KIND = {"d": LevelKind.DENSE, "s": LevelKind.COMPRESSED}


def build_intersection_cin(fmt, dtype=torch.float32, *, commuted=False):
    """C = A * B elementwise over one shared format (the einsum shape)."""

    rank = len(fmt)
    ivars = tuple(IndexVar(name) for name in "ijkl"[:rank])
    a = TensorVar("A", fmt=fmt, dtype=dtype)
    b = TensorVar("B", fmt=fmt, dtype=dtype)
    c = TensorVar("C", fmt=fmt, dtype=dtype)
    left, right = (b, a) if commuted else (a, b)
    stmt = TensorAssign(c[ivars], CINBinaryOp(Operation.MUL, left[ivars], right[ivars]))
    for index_var in reversed(ivars):
        stmt = ForAll(index_var, stmt)
    return stmt


_SHAPES = {
    "ss": (4, 5),
    "sss": (3, 4, 5),
    "dss": (3, 4, 5),
    "ssss": (2, 3, 4, 5),
}


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


def validated_storage_pieces(result, fmt, shape):
    """Assert honest identity-ordered storage; return per-level pieces."""

    mode_indices = result.storage.index.mode_indices
    assert len(mode_indices) == len(fmt)
    pieces = []
    parents = 1
    for level, kind in enumerate(fmt):
        if kind == "d":
            assert list(mode_indices[level]) == []
            parents = parents * shape[level]
            pieces.append(None)
            continue
        pos = mode_indices[level][0].tolist()
        crd = mode_indices[level][1].tolist()
        assert len(pos) == parents + 1
        assert pos[0] == 0
        assert pos == sorted(pos)
        assert pos[-1] == len(crd)
        for segment in range(parents):
            entries = crd[pos[segment] : pos[segment + 1]]
            assert entries == sorted(set(entries))
            assert all(0 <= entry < shape[level] for entry in entries)
        parents = len(crd)
        pieces.append((pos, crd))
    values = result.storage.value.tolist()
    assert len(values) == parents
    return pieces, values


# -- source parity -----------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fmt", ["ss", "sss", "dss", "ssss"])
def test_source_parity_matches_legacy_in_both_arms(fmt, dtype):
    shape = _SHAPES[fmt]
    bindings = ((shape, dtype), (shape, dtype))
    for arm in (False, True):
        comparison = compare_generated_sources(
            build_intersection_cin(fmt, dtype),
            shape,
            bindings,
            compile_options=auto_options(arm),
        )
        assert comparison.identical, f"fmt {fmt} arm regblock={arm} diverged"


@pytest.mark.parametrize("fmt", ["ss", "sss"])
def test_commuted_operands_hold_parity(fmt):
    shape = _SHAPES[fmt]
    bindings = ((shape, torch.float32), (shape, torch.float32))
    for arm in (False, True):
        comparison = compare_generated_sources(
            build_intersection_cin(fmt, commuted=True),
            shape,
            bindings,
            compile_options=auto_options(arm),
        )
        assert comparison.identical


def test_family_is_serial_and_route_stable():
    """The assembly family emits no OpenMP marking, in either arm."""

    sources = set()
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            build_intersection_cin("ss"),
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        assert "#pragma omp" not in kernel.cpp_source
        assert "C0_crd.push_back" in kernel.cpp_source
        sources.add(kernel.cpp_source)
        replay = compile_cin_via_loopir(
            build_intersection_cin("ss"),
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        assert replay.cpp_source in sources


def test_canonical_dump_is_arm_stable_and_erases_to_base():
    dumps = []
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            build_intersection_cin("sss"),
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
    assert '"kind":"append_entry"' in dumps[0]


# -- compiled execution ------------------------------------------------------


def _fixture(fmt, dtype=torch.float32, seed=20260730):
    torch.manual_seed(seed)
    shape = _SHAPES[fmt]
    dense_a = ((torch.rand(shape) < 0.35) * torch.randn(shape)).to(dtype)
    dense_b = ((torch.rand(shape) < 0.35) * torch.randn(shape)).to(dtype)
    return dense_a, dense_b


@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize("fmt", ["ss", "sss", "dss"])
def test_compiled_execution_matches_every_reference(regblock_enabled, fmt):
    dense_a, dense_b = _fixture(fmt)
    shape = _SHAPES[fmt]
    cin = build_intersection_cin(fmt)
    result = executed(
        cin,
        shape,
        (sparse(dense_a, "A", fmt), sparse(dense_b, "B", fmt)),
        regblock_enabled,
    )
    # Validate the storage before to_torch(): densification converts the
    # STensor's layout in place.
    pieces, values = validated_storage_pieces(result, fmt, shape)
    assert torch.allclose(result.to_torch(), dense_a * dense_b, atol=1e-3, rtol=1e-3)

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


def test_rank4_execution_matches_reference():
    dense_a, dense_b = _fixture("ssss")
    result = executed(
        build_intersection_cin("ssss"),
        _SHAPES["ssss"],
        (sparse(dense_a, "A", "ssss"), sparse(dense_b, "B", "ssss")),
        False,
    )
    assert torch.allclose(result.to_torch(), dense_a * dense_b, atol=1e-3, rtol=1e-3)


def test_float64_execution_matches_reference():
    dense_a, dense_b = _fixture("ss", torch.float64)
    result = executed(
        build_intersection_cin("ss", torch.float64),
        (4, 5),
        (sparse(dense_a, "A", "ss"), sparse(dense_b, "B", "ss")),
        False,
    )
    assert torch.allclose(result.to_torch(), dense_a * dense_b, atol=1e-9, rtol=1e-9)


def test_empty_intermediate_parents_cascade():
    """Structurally intersecting parents with empty child merges vanish."""

    dense_a = torch.tensor(
        [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    dense_b = torch.tensor(
        [[0.0, 5.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 7.0], [2.0, 0.0, 0.0]]
    )
    result = executed(
        build_intersection_cin("ss"),
        (4, 3),
        (sparse(dense_a, "A", "ss"), sparse(dense_b, "B", "ss")),
        False,
    )
    mode_indices = result.storage.index.mode_indices
    assert mode_indices[0][0].tolist() == [0, 1]
    assert mode_indices[0][1].tolist() == [3]
    assert mode_indices[1][0].tolist() == [0, 1]
    assert mode_indices[1][1].tolist() == [0]
    assert result.storage.value.tolist() == [2.0]


def test_rank3_empty_middle_parent_is_suppressed():
    """A stored (i, j) pair whose leaf merge is empty appends no parent."""

    shape = (2, 2, 3)
    dense_a = torch.zeros(shape)
    dense_b = torch.zeros(shape)
    dense_a[0, 0, 0] = 1.0
    dense_b[0, 0, 1] = 2.0  # leaf-disjoint at the shared (0, 0) parent
    dense_a[1, 1, 2] = 3.0
    dense_b[1, 1, 2] = 4.0
    result = executed(
        build_intersection_cin("sss"),
        shape,
        (sparse(dense_a, "A", "sss"), sparse(dense_b, "B", "sss")),
        False,
    )
    mode_indices = result.storage.index.mode_indices
    assert mode_indices[0][1].tolist() == [1]
    assert mode_indices[1][1].tolist() == [1]
    assert mode_indices[2][1].tolist() == [2]
    assert result.storage.value.tolist() == [12.0]


def test_disjoint_and_empty_operands_produce_canonical_empty_storage():
    dense_a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    dense_b = torch.tensor([[0.0, 3.0], [4.0, 0.0]])
    result = executed(
        build_intersection_cin("ss"),
        (2, 2),
        (sparse(dense_a, "A", "ss"), sparse(dense_b, "B", "ss")),
        False,
    )
    assert result.storage.index.mode_indices[0][0].tolist() == [0, 0]
    assert result.storage.value.tolist() == []

    empty = torch.zeros(2, 2)
    result = executed(
        build_intersection_cin("ss"),
        (2, 2),
        (sparse(empty, "A", "ss"), sparse(dense_b, "B", "ss")),
        False,
    )
    assert result.storage.index.mode_indices[0][0].tolist() == [0, 0]
    assert result.storage.value.tolist() == []


def test_zero_extent_cells_execute():
    dense = torch.zeros(0, 3)
    result = executed(
        build_intersection_cin("ss"),
        (0, 3),
        (sparse(dense, "A", "ss"), sparse(dense, "B", "ss")),
        False,
    )
    assert result.storage.index.mode_indices[0][0].tolist() == [0, 0]
    assert result.storage.value.tolist() == []


def test_explicit_zero_products_are_retained():
    """A stored zero times a stored value appends an explicit zero."""

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
    dense_b[1, 2] = 5.0
    result = executed(
        build_intersection_cin("ss"),
        (3, 4),
        (explicit_zero, sparse(dense_b, "B", "ss")),
        False,
    )
    assert result.storage.index.mode_indices[0][1].tolist() == [1]
    assert result.storage.index.mode_indices[1][1].tolist() == [2]
    assert result.storage.value.tolist() == [0.0]


def test_deterministic_storage_and_repeated_execution():
    dense_a, dense_b = _fixture("sss", seed=7)
    snapshots = []
    for _ in range(3):
        result = executed(
            build_intersection_cin("sss"),
            _SHAPES["sss"],
            (sparse(dense_a, "A", "sss"), sparse(dense_b, "B", "sss")),
            False,
        )
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


def test_execution_matches_the_public_einsum_differential():
    """scorch.einsum('ij,ij->ij', ss, ss) is the named production caller."""

    import scorch

    dense_a, dense_b = _fixture("ss", seed=11)
    public = scorch.einsum(
        "ij,ij->ij", sparse(dense_a, "A", "ss"), sparse(dense_b, "B", "ss")
    )
    public_dense = (
        public.to_torch() if isinstance(public, STensor) else torch.as_tensor(public)
    )
    for _ in range(3):
        result = executed(
            build_intersection_cin("ss"),
            (4, 5),
            (sparse(dense_a, "A", "ss"), sparse(dense_b, "B", "ss")),
            False,
        )
        assert torch.allclose(result.to_torch(), public_dense, atol=1e-3, rtol=1e-3)


# -- hand-built target programs ----------------------------------------------


def build_multi_compressed_program(*, forge_positions=False):
    """The exact ss*ss->ss LoopIR program the pipeline lowers, hand-built."""

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
    leaf = builder.append_entry(
        c,
        (builder.index_value(index_i), builder.index_value(index_j)),
        builder.binary(
            LoopIRBinaryOp.MUL,
            builder.cursor_value(cursor_a1.cursor),
            builder.cursor_value(cursor_b1.cursor),
        ),
    )
    inner_positions = (
        ()
        if forge_positions
        else (builder.new_position_id(), builder.new_position_id())
    )
    inner = builder.merged_sparse_for(
        MergeMode.INTERSECTION,
        (cursor_a1, cursor_b1),
        index_j,
        builder.block((leaf,)),
        inner_positions,
    )
    outer = builder.merged_sparse_for(
        MergeMode.INTERSECTION,
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
    return program, (a, b, c)


def test_hand_built_program_lowers_and_matches_pipeline_source():
    program, (a, b, _) = build_multi_compressed_program()
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
        build_intersection_cin("ss"),
        (4, 5),
        (((4, 5), torch.float32), ((4, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    assert cpp_source == kernel.cpp_source


def test_unbound_merge_positions_fail_closed():
    program, (a, b, _) = build_multi_compressed_program(forge_positions=True)
    options = CompileOptions.from_environment(environ={})
    context = CompilationContext(options)
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes={a: (4, 5), b: (4, 5)},
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=context,
        )
    assert error.value.defect.code == "unsupported_program_shape"
    assert context._failed_stage_id is CompilerStageId.LOOPIR_TO_LLIR_LOWERING


# -- target-owned checked output mutations ---------------------------------


def _compile_b3_target(fmt="ss"):
    shape = _SHAPES[fmt]
    return compile_cin_via_loopir(
        build_intersection_cin(fmt),
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(False),
    )


def _b3_output_mutation_census(value):
    """Return unchecked output assignments and direct safe mutation names."""

    from scorch.compiler import llir

    unchecked = []
    direct_calls = []
    pending = [value]
    visited = set()
    while pending:
        current = pending.pop()
        if isinstance(current, (llir.Node, list, tuple)):
            identity = id(current)
            if identity in visited:
                continue
            visited.add(identity)
        if type(current) is llir.Assign:
            target = current.var
            if (
                type(target) is llir.ArrayAccess
                and type(target.array) is llir.Var
                and (
                    target.array.name == "C_values"
                    or (
                        target.array.name.startswith("C")
                        and target.array.name.endswith(("_pos", "_crd"))
                    )
                )
            ):
                unchecked.append(current)
        if type(current) is llir.FunctionCallStmt:
            if current.name == "scorch_vector_set" or current.name.startswith("C"):
                direct_calls.append(current.name)
        if isinstance(current, llir.Node):
            pending.extend(vars(current).values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
    return unchecked, direct_calls


def test_b3_enters_dynamic_vector_pass_with_only_checked_output_mutations(
    monkeypatch,
):
    """The complete rank-four target is safe before the generic rewrite."""

    import scorch.compiler.llir_pass_manager as pass_manager

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
    _compile_b3_target("ssss")
    assert len(censuses) == 1
    unchecked, calls = censuses[0]
    assert unchecked == []
    assert calls.count("C_values.emplace_back") == 1
    assert calls.count("C3_crd.emplace_back") == 1
    assert calls.count("scorch_vector_set") == 8


def test_b3_dynamic_vector_pass_still_runs_but_is_byte_neutral(monkeypatch):
    """Omitting the now-redundant rewrite leaves byte-exact safe output."""

    import scorch.compiler.llir_pass_manager as pass_manager

    baseline = _compile_b3_target("ssss")
    calls = []

    def omit(value, context):
        calls.append((value, context))
        return value

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        omit,
    )
    omitted = _compile_b3_target("ssss")
    assert len(calls) == 1
    assert omitted.cpp_source == baseline.cpp_source


_CHECKED_MUTATION_CENSUS = {
    # fmt -> (leaf level, parent push_backs, scorch_vector_set count)
    "ss": (1, ("C0_crd.push_back",), 4),
    "sss": (2, ("C0_crd.push_back", "C1_crd.push_back"), 6),
    "dss": (2, ("C1_crd.push_back",), 5),
}


@pytest.mark.parametrize("fmt", ["ss", "sss", "dss"])
def test_b3_checked_mutation_census_holds_across_formats(fmt, monkeypatch):
    """Every B3 format is safe before the generic rewrite, not only rank 4."""

    import scorch.compiler.llir_pass_manager as pass_manager

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
    _compile_b3_target(fmt)
    assert len(censuses) == 1
    unchecked, calls = censuses[0]
    assert unchecked == []
    leaf_level, parent_pushes, vector_sets = _CHECKED_MUTATION_CENSUS[fmt]
    assert calls.count("C_values.emplace_back") == 1
    assert calls.count(f"C{leaf_level}_crd.emplace_back") == 1
    for parent_push in parent_pushes:
        assert calls.count(parent_push) == 1
    assert calls.count("scorch_vector_set") == vector_sets


@pytest.mark.parametrize("fmt", ["ss", "dss"])
def test_b3_dynamic_vector_pass_neutrality_holds_across_formats(fmt, monkeypatch):
    """The no-op rewrite leaves byte-exact source for rank-2 and dense-prefix."""

    import scorch.compiler.llir_pass_manager as pass_manager

    baseline = _compile_b3_target(fmt)

    def omit(value, context):
        return value

    monkeypatch.setattr(
        pass_manager,
        "rewrite_dynamic_vector_accesses",
        omit,
    )
    omitted = _compile_b3_target(fmt)
    assert omitted.cpp_source == baseline.cpp_source


# -- adjacent seams stay fail-closed -----------------------------------------


def _seam_cells():
    i, j = IndexVar("i"), IndexVar("j")
    i4, j4, k4, l4 = (IndexVar(name) for name in "ijkl")
    ttm = ForAll(
        i4,
        ForAll(
            j4,
            ForAll(
                k4,
                ForAll(
                    l4,
                    TensorAssign(
                        TensorVar("C", fmt="sss")[i4, j4, l4],
                        CINBinaryOp(
                            Operation.MUL,
                            TensorVar("A", fmt="sss")[i4, j4, k4],
                            TensorVar("B", fmt="dd")[k4, l4],
                        ),
                        op=Operation.ADD,
                    ),
                ),
            ),
        ),
    )
    interleaved = build_intersection_cin("sds")
    three = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("C", fmt="ss")[i, j],
                CINBinaryOp(
                    Operation.MUL,
                    CINBinaryOp(
                        Operation.MUL,
                        TensorVar("A", fmt="ss")[i, j],
                        TensorVar("B", fmt="ss")[i, j],
                    ),
                    TensorVar("D", fmt="ss")[i, j],
                ),
            ),
        ),
    )
    f32 = torch.float32
    return [
        (
            "ttm_reduction",
            ttm,
            (3, 4, 5),
            (((3, 4, 6), f32), ((6, 5), f32)),
            "unsupported_sparse_output",
        ),
        # Recorded seam move: the two-dense-prefix ``ddss`` intersection that
        # used to sit here is the admitted flattened-prefix family
        # (``test_loopir_multi_dense_prefix_target.py``).  Its neighbour on
        # the same seam -- an INTERLEAVED result whose compressed level sits
        # ABOVE a dense one, so the dense level's parent count is a dynamic
        # stored-coordinate count rather than a static product -- keeps the
        # exact code.
        (
            "interleaved_dense_between_compressed",
            interleaved,
            (3, 4, 5),
            (((3, 4, 5), f32), ((3, 4, 5), f32)),
            "unsupported_sparse_output",
        ),
        (
            "three_operand_chain",
            three,
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


def test_b1_reduction_route_is_untouched():
    """ss@ss->ss with the ADD update stays the B1 workspace family."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    b1 = ForAll(
        i,
        ForAll(
            k,
            ForAll(
                j,
                TensorAssign(
                    TensorVar("C", fmt="ss")[i, j],
                    CINBinaryOp(
                        Operation.MUL,
                        TensorVar("A", fmt="ss")[i, k],
                        TensorVar("B", fmt="ss")[k, j],
                    ),
                    op=Operation.ADD,
                ),
            ),
        ),
    )
    for arm in (False, True):
        comparison = compare_generated_sources(
            b1,
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        assert comparison.identical
