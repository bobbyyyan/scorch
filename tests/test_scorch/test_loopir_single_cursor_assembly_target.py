"""Phase-7 single-cursor multi-compressed assembly: the ordered-stream family.

Copy and elementwise chains whose dense-prefix/multi-compressed-suffix
results are assembled from single-cursor stored streams — ``ss``/``sss``/
``dss``/``ssss``/``dsss`` copies (the formats production ``einsum`` infers
for identity expressions over compressed inputs) and MUL against all-dense
or partly-dense co-operands (``ss*dd``, ``ss*ds``, ``dss*ddd``) whose
suffix mixes single-cursor and two-cursor INTERSECTION levels.  Each
compressed result level appends its stored entries in order behind one
conditional parent append; single-operand assemblies pre-size dense-parent
position vectors exactly as the legacy assembler does.

The legacy comparand is honest for this family (production ``einsum``
executes it for these shapes and matches the dense reference with
well-formed storage), so the gate is byte parity with
``legacy_generated_cpp`` in both automatic policy arms plus the production
LoopIR oracle and the PyTorch dense reference — the B1/B3 discipline.
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
from scorch.compiler.loopir.nodes import LevelKind, ScalarType
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


def build_stream_cin(result_fmt, operand_fmts, dtype=torch.float32):
    """C = A (or A * B) over one shared index tuple, formats per operand."""

    rank = len(result_fmt)
    ivars = tuple(IndexVar(name) for name in "ijkl"[:rank])
    result = TensorVar("C", fmt=result_fmt, dtype=dtype)
    accesses = [
        TensorVar(name, fmt=fmt, dtype=dtype)[ivars]
        for name, fmt in zip("AB", operand_fmts)
    ]
    rhs = accesses[0]
    for access in accesses[1:]:
        rhs = CINBinaryOp(Operation.MUL, rhs, access)
    stmt = TensorAssign(result[ivars], rhs)
    for index_var in reversed(ivars):
        stmt = ForAll(index_var, stmt)
    return stmt


def sparse(dense, name, fmt):
    return STensor.from_torch(dense.clone(), name).to_sparse(fmt)


def operand_stensor(dense, name, fmt):
    if all(ch == "d" for ch in fmt):
        return STensor.from_torch(dense.clone(), name)
    return sparse(dense, name, fmt)


def executed(cin, result_shape, stensors, regblock_enabled):
    out = execute_cin_via_loopir(
        cin,
        result_shape,
        *stensors,
        compile_options=auto_options(regblock_enabled, jit=True),
    )
    return out[0] if isinstance(out, tuple) else out


def oracle_storage(kernel, dense_inputs, operand_fmts, result_shape):
    """Base and scheduled oracle agreement; returns the level storage."""

    lowering = kernel.lowering
    inputs = {}
    for symbol, dense, fmt in zip(
        lowering.rhs_access_symbols, dense_inputs, operand_fmts
    ):
        if all(ch == "d" for ch in fmt):
            inputs[symbol] = dense.tolist()
        else:
            inputs[symbol] = LevelTensorStorage.from_dense(
                dense.tolist(),
                tuple(dense.shape),
                tuple(range(len(fmt))),
                tuple(_KIND[ch] for ch in fmt),
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


def _fixture(fmt, dtype=torch.float32, seed=20260731, density=0.4):
    torch.manual_seed(seed)
    shape = _SHAPES[fmt]
    return ((torch.rand(shape) < density) * torch.randn(shape)).to(dtype)


# -- source parity -----------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fmt", ["ss", "sss", "dss", "ssss", "dsss"])
def test_copy_source_parity_matches_legacy_in_both_arms(fmt, dtype):
    shape = _SHAPES[fmt]
    for arm in (False, True):
        comparison = compare_generated_sources(
            build_stream_cin(fmt, (fmt,), dtype),
            shape,
            ((shape, dtype),),
            compile_options=auto_options(arm),
        )
        assert comparison.identical, f"copy {fmt} arm regblock={arm} diverged"


@pytest.mark.parametrize(
    "result_fmt,operand_fmts",
    [
        ("ss", ("ss", "dd")),
        ("ss", ("dd", "ss")),
        ("ss", ("ss", "ds")),
        ("ss", ("ds", "ss")),
        ("dss", ("dss", "ddd")),
        ("sss", ("sss", "ddd")),
    ],
    ids=["ss*dd", "dd*ss", "ss*ds", "ds*ss", "dss*ddd", "sss*ddd"],
)
def test_elementwise_source_parity_matches_legacy_in_both_arms(
    result_fmt, operand_fmts
):
    shape = _SHAPES[result_fmt]
    bindings = tuple((shape, torch.float32) for _ in operand_fmts)
    for arm in (False, True):
        comparison = compare_generated_sources(
            build_stream_cin(result_fmt, operand_fmts),
            shape,
            bindings,
            compile_options=auto_options(arm),
        )
        assert comparison.identical, f"{operand_fmts} arm regblock={arm} diverged"


def test_family_is_serial_and_route_stable():
    """The stream family emits no OpenMP marking, in either arm."""

    sources = set()
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            build_stream_cin("ss", ("ss",)),
            (4, 5),
            (((4, 5), torch.float32),),
            compile_options=auto_options(arm),
        )
        assert "#pragma omp" not in kernel.cpp_source
        assert "C0_crd.push_back" in kernel.cpp_source
        assert "C_values.emplace_back" in kernel.cpp_source
        sources.add(kernel.cpp_source)
        replay = compile_cin_via_loopir(
            build_stream_cin("ss", ("ss",)),
            (4, 5),
            (((4, 5), torch.float32),),
            compile_options=auto_options(arm),
        )
        assert replay.cpp_source in sources


def test_single_operand_copy_pre_sizes_dense_parent_positions():
    """The dss copy sizes C1_pos from the dense row extent, like legacy."""

    kernel = compile_cin_via_loopir(
        build_stream_cin("dss", ("dss",)),
        (3, 4, 5),
        (((3, 4, 5), torch.float32),),
        compile_options=auto_options(False),
    )
    assert "std::vector<int> C1_pos((size_t)C0_size + 1, 0);" in kernel.cpp_source
    assert "scorch_vector_set(C2_pos, 0, 0);" in kernel.cpp_source


def test_two_operand_elementwise_keeps_dynamic_positions():
    """Multi-operand assemblies keep the dynamic checked position form."""

    kernel = compile_cin_via_loopir(
        build_stream_cin("dss", ("dss", "ddd")),
        (3, 4, 5),
        (((3, 4, 5), torch.float32), ((3, 4, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    assert "std::vector<int> C1_pos;" in kernel.cpp_source
    assert "scorch_vector_set(C1_pos, 0, 0);" in kernel.cpp_source


def test_dense_co_operand_keeps_the_legacy_prefetch():
    """The mixed sparse-by-dense stream retains the legacy prefetch hint."""

    kernel = compile_cin_via_loopir(
        build_stream_cin("ss", ("ss", "dd")),
        (4, 5),
        (((4, 5), torch.float32), ((4, 5), torch.float32)),
        compile_options=auto_options(False),
    )
    assert "__builtin_prefetch" in kernel.cpp_source


def test_canonical_dump_is_arm_stable_and_erases_to_base():
    dumps = []
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            build_stream_cin("dss", ("dss",)),
            (3, 4, 5),
            (((3, 4, 5), torch.float32),),
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
    assert '"kind":"sparse_for"' in dumps[0]


def test_compile_is_deterministic_per_arm():
    for arm in (False, True):
        first = compile_cin_via_loopir(
            build_stream_cin("ss", ("ss", "dd")),
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        second = compile_cin_via_loopir(
            build_stream_cin("ss", ("ss", "dd")),
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        assert first.cpp_source == second.cpp_source
        assert first.program_dump == second.program_dump


# -- compiled execution ------------------------------------------------------


@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize("fmt", ["ss", "dss"])
def test_compiled_copy_matches_every_reference(regblock_enabled, fmt):
    dense = _fixture(fmt)
    shape = _SHAPES[fmt]
    cin = build_stream_cin(fmt, (fmt,))
    result = executed(cin, shape, (sparse(dense, "A", fmt),), regblock_enabled)
    pieces, values = validated_storage_pieces(result, fmt, shape)
    assert torch.allclose(result.to_torch(), dense, atol=1e-3, rtol=1e-3)

    kernel = compile_cin_via_loopir(
        cin,
        shape,
        ((shape, torch.float32),),
        compile_options=auto_options(regblock_enabled),
    )
    oracle = oracle_storage(kernel, (dense,), (fmt,), shape)
    for level, ch in enumerate(fmt):
        if ch != "s":
            continue
        pos, crd = pieces[level]
        assert tuple(pos) == oracle.positions[level]
        assert tuple(crd) == oracle.coordinates[level]
    assert len(values) == len(oracle.values)
    for got, expected in zip(values, oracle.values):
        assert got == pytest.approx(expected, abs=1e-5, rel=1e-5)


@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize(
    "operand_fmts",
    [("ss", "dd"), ("dd", "ss"), ("ss", "ds")],
    ids=["ss*dd", "dd*ss", "ss*ds"],
)
def test_compiled_elementwise_matches_every_reference(regblock_enabled, operand_fmts):
    shape = _SHAPES["ss"]
    torch.manual_seed(3)
    dense_a = (torch.rand(shape) < 0.4) * torch.randn(shape)
    dense_b = torch.randn(shape)
    cin = build_stream_cin("ss", operand_fmts)
    stensors = tuple(
        operand_stensor(dense, name, fmt)
        for dense, name, fmt in zip((dense_a, dense_b), "AB", operand_fmts)
    )
    result = executed(cin, shape, stensors, regblock_enabled)
    pieces, values = validated_storage_pieces(result, "ss", shape)
    assert torch.allclose(result.to_torch(), dense_a * dense_b, atol=1e-3, rtol=1e-3)

    kernel = compile_cin_via_loopir(
        cin,
        shape,
        ((shape, torch.float32), (shape, torch.float32)),
        compile_options=auto_options(regblock_enabled),
    )
    oracle = oracle_storage(kernel, (dense_a, dense_b), operand_fmts, shape)
    for level in range(2):
        pos, crd = pieces[level]
        assert tuple(pos) == oracle.positions[level]
        assert tuple(crd) == oracle.coordinates[level]
    assert len(values) == len(oracle.values)
    for got, expected in zip(values, oracle.values):
        assert got == pytest.approx(expected, abs=1e-5, rel=1e-5)


def test_rank4_copies_execute():
    for fmt in ("ssss", "dsss"):
        dense = _fixture(fmt)
        result = executed(
            build_stream_cin(fmt, (fmt,)),
            _SHAPES[fmt],
            (sparse(dense, "A", fmt),),
            False,
        )
        validated_storage_pieces(result, fmt, _SHAPES[fmt])
        assert torch.allclose(result.to_torch(), dense, atol=1e-3, rtol=1e-3)


def test_float64_execution_matches_reference():
    dense = _fixture("ss", torch.float64)
    result = executed(
        build_stream_cin("ss", ("ss",), torch.float64),
        (4, 5),
        (sparse(dense, "A", "ss"),),
        False,
    )
    assert torch.allclose(result.to_torch(), dense, atol=1e-9, rtol=1e-9)


def test_copy_preserves_stored_structure_exactly():
    """The copy's storage mirrors the operand's stored streams one-to-one."""

    dense = _fixture("sss", seed=5)
    operand = sparse(dense, "A", "sss")
    operand_modes = [
        [tensor.tolist() for tensor in level]
        for level in operand.storage.index.mode_indices
    ]
    result = executed(
        build_stream_cin("sss", ("sss",)),
        _SHAPES["sss"],
        (operand,),
        False,
    )
    result_modes = [
        [tensor.tolist() for tensor in level]
        for level in result.storage.index.mode_indices
    ]
    assert result_modes == operand_modes
    assert result.storage.value.tolist() == operand.storage.value.tolist()


def test_explicit_zero_dense_factors_are_retained():
    """A stored entry times a dense zero appends an explicit zero."""

    dense_a = torch.zeros(3, 4)
    dense_a[1, 2] = 5.0
    dense_a[2, 0] = 3.0
    dense_b = torch.ones(3, 4)
    dense_b[1, 2] = 0.0
    result = executed(
        build_stream_cin("ss", ("ss", "dd")),
        (3, 4),
        (sparse(dense_a, "A", "ss"), STensor.from_torch(dense_b.clone(), "B")),
        False,
    )
    assert result.storage.index.mode_indices[0][1].tolist() == [1, 2]
    assert result.storage.index.mode_indices[1][1].tolist() == [2, 0]
    assert result.storage.value.tolist() == [0.0, 3.0]


def test_explicit_zero_operand_entries_are_copied():
    """A hand-built stored zero survives the copy as a stored zero."""

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
    result = executed(
        build_stream_cin("ss", ("ss",)),
        (3, 4),
        (explicit_zero,),
        False,
    )
    assert result.storage.index.mode_indices[0][1].tolist() == [1]
    assert result.storage.index.mode_indices[1][1].tolist() == [2]
    assert result.storage.value.tolist() == [0.0]


def test_empty_intermediate_parents_are_suppressed():
    """A hand-built parent with an empty child segment appends nothing."""

    pos0 = torch.tensor([0, 2], dtype=torch.int32)
    crd0 = torch.tensor([0, 2], dtype=torch.int32)
    pos1 = torch.tensor([0, 0, 1], dtype=torch.int32)  # row 0 stored empty
    crd1 = torch.tensor([3], dtype=torch.int32)
    operand = STensor(
        name="A",
        shape=(3, 4),
        index=TensorIndex("ss", [[pos0, crd0], [pos1, crd1]]),
        value=torch.tensor([7.0]),
    )
    result = executed(
        build_stream_cin("ss", ("ss",)),
        (3, 4),
        (operand,),
        False,
    )
    assert result.storage.index.mode_indices[0][0].tolist() == [0, 1]
    assert result.storage.index.mode_indices[0][1].tolist() == [2]
    assert result.storage.index.mode_indices[1][0].tolist() == [0, 1]
    assert result.storage.index.mode_indices[1][1].tolist() == [3]
    assert result.storage.value.tolist() == [7.0]


def test_empty_rows_and_empty_operands_produce_canonical_storage():
    dense = torch.zeros(3, 4)
    dense[1, 2] = 2.0
    result = executed(
        build_stream_cin("ss", ("ss",)),
        (3, 4),
        (sparse(dense, "A", "ss"),),
        False,
    )
    assert result.storage.index.mode_indices[0][0].tolist() == [0, 1]
    assert result.storage.index.mode_indices[0][1].tolist() == [1]

    empty = torch.zeros(3, 4)
    result = executed(
        build_stream_cin("ss", ("ss",)),
        (3, 4),
        (sparse(empty, "A", "ss"),),
        False,
    )
    assert result.storage.index.mode_indices[0][0].tolist() == [0, 0]
    assert result.storage.value.tolist() == []


def test_dense_prefix_empty_rows_keep_full_position_arrays():
    """The pre-sized dss copy carries every dense row's position slot."""

    dense = torch.zeros(3, 4, 5)
    dense[2, 1, 3] = 4.0
    result = executed(
        build_stream_cin("dss", ("dss",)),
        (3, 4, 5),
        (sparse(dense, "A", "dss"),),
        False,
    )
    mode_indices = result.storage.index.mode_indices
    assert mode_indices[1][0].tolist() == [0, 0, 0, 1]
    assert mode_indices[1][1].tolist() == [1]
    assert mode_indices[2][0].tolist() == [0, 1]
    assert mode_indices[2][1].tolist() == [3]
    assert result.storage.value.tolist() == [4.0]


def test_zero_extent_cells_execute():
    dense = torch.zeros(0, 3)
    result = executed(
        build_stream_cin("ss", ("ss",)),
        (0, 3),
        (sparse(dense, "A", "ss"),),
        False,
    )
    assert result.storage.index.mode_indices[0][0].tolist() == [0, 0]
    assert result.storage.value.tolist() == []


def test_deterministic_storage_and_repeated_execution():
    dense = _fixture("dss", seed=7)
    snapshots = []
    for _ in range(3):
        result = executed(
            build_stream_cin("dss", ("dss",)),
            _SHAPES["dss"],
            (sparse(dense, "A", "dss"),),
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


@pytest.mark.parametrize(
    "expression,fmt,operand_count",
    [
        ("ij->ij", "ss", 1),
        ("ijk->ijk", "dss", 1),
        ("ij,ij->ij", "ss", 2),
    ],
    ids=["ss-copy", "dss-copy", "ss-mul-dd"],
)
def test_execution_matches_the_public_einsum_differential(
    expression, fmt, operand_count
):
    """Identity and mixed-MUL einsum are the named production callers."""

    import scorch

    shape = _SHAPES[fmt]
    torch.manual_seed(11)
    dense_a = (torch.rand(shape) < 0.4) * torch.randn(shape)
    if operand_count == 1:
        tensors = (sparse(dense_a, "A", fmt),)
        operand_fmts = (fmt,)
        reference = dense_a
        stensors = (sparse(dense_a, "A", fmt),)
    else:
        dense_b = torch.randn(shape)
        tensors = (
            sparse(dense_a, "A", fmt),
            STensor.from_torch(dense_b.clone(), "B"),
        )
        operand_fmts = (fmt, "d" * len(fmt))
        reference = dense_a * dense_b
        stensors = (
            sparse(dense_a, "A", fmt),
            STensor.from_torch(dense_b.clone(), "B"),
        )
    public = scorch.einsum(expression, *tensors)
    public_dense = (
        public.to_torch() if isinstance(public, STensor) else torch.as_tensor(public)
    )
    assert torch.allclose(public_dense, reference, atol=1e-3, rtol=1e-3)
    for _ in range(3):
        result = executed(
            build_stream_cin(fmt, operand_fmts),
            shape,
            stensors,
            False,
        )
        assert torch.allclose(result.to_torch(), public_dense, atol=1e-3, rtol=1e-3)


# -- target-owned checked output mutations -----------------------------------


def test_stream_family_enters_dynamic_pass_with_only_checked_mutations(
    monkeypatch,
):
    """Dynamic result vectors see only checked mutations before the rewrite.

    The pre-sized dense-parent position vector is written through bounded
    indexed assignments (its extent is ABI-validated), exactly like the
    legacy emission; every dynamically grown vector sees only checked
    appends and ``scorch_vector_set`` closes.
    """

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
    compile_cin_via_loopir(
        build_stream_cin("dss", ("dss",)),
        (3, 4, 5),
        (((3, 4, 5), torch.float32),),
        compile_options=auto_options(False),
    )
    assert len(censuses) == 1
    unchecked, calls = censuses[0]
    # The only indexed stores target the pre-sized C1_pos vector.
    assert {stmt.var.array.name for stmt in unchecked} <= {"C1_pos"}
    assert calls.count("C_values.emplace_back") == 1
    assert calls.count("C2_crd.emplace_back") == 1
    assert calls.count("C1_crd.push_back") == 1
    assert calls.count("scorch_vector_set") == 2


def test_stream_family_dynamic_pass_is_byte_neutral(monkeypatch):
    """Omitting the generic rewrite leaves byte-exact safe source."""

    import scorch.compiler.llir_pass_manager as pass_manager

    baseline = compile_cin_via_loopir(
        build_stream_cin("dss", ("dss",)),
        (3, 4, 5),
        (((3, 4, 5), torch.float32),),
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
        build_stream_cin("dss", ("dss",)),
        (3, 4, 5),
        (((3, 4, 5), torch.float32),),
        compile_options=auto_options(False),
    )
    assert omitted.cpp_source == baseline.cpp_source


# -- hand-built target programs ----------------------------------------------


def build_single_cursor_copy_program():
    """The exact ss->ss copy LoopIR program the pipeline lowers, hand-built."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, c = (builder.new_symbol_id() for _ in range(2))
    cc_levels = lambda: (  # noqa: E731
        builder.level(LevelKind.COMPRESSED, 0),
        builder.level(LevelKind.COMPRESSED, 1),
    )
    decl_a = builder.tensor(
        a, "A", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), cc_levels()
    )
    decl_c = builder.tensor(
        c, "C", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), cc_levels()
    )
    index_i = builder.new_index_id()
    index_j = builder.new_index_id()
    position_a0 = builder.new_position_id()
    position_a1 = builder.new_position_id()
    cursor_a0 = builder.sparse_cursor(
        builder.new_cursor_id(), a, 0, builder.root_position()
    )
    cursor_a1 = builder.sparse_cursor(
        builder.new_cursor_id(), a, 1, builder.position_value(position_a0)
    )
    leaf = builder.append_entry(
        c,
        (builder.index_value(index_i), builder.index_value(index_j)),
        builder.cursor_value(cursor_a1.cursor),
    )
    inner = builder.sparse_for(cursor_a1, position_a1, index_j, builder.block((leaf,)))
    outer = builder.sparse_for(cursor_a0, position_a0, index_i, builder.block((inner,)))
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_c),
        (a,),
        (c,),
        builder.block((outer,)),
    )
    return program, (a, c)


def test_hand_built_program_lowers_and_matches_pipeline_source():
    program, (a, _) = build_single_cursor_copy_program()
    options = CompileOptions.from_environment(environ={})
    context = CompilationContext(options)
    function = lower_loopir_to_llir(
        program,
        input_shapes={a: (4, 5)},
        result_shape=(4, 5),
        compile_options=options,
        compilation_context=context,
    )
    from scorch.ops import _lower_generated_llir

    cpp_source = _lower_generated_llir(function, options, context)
    kernel = compile_cin_via_loopir(
        build_stream_cin("ss", ("ss",)),
        (4, 5),
        (((4, 5), torch.float32),),
        compile_options=auto_options(False),
    )
    assert cpp_source == kernel.cpp_source


def test_hand_built_program_executes_on_the_oracle():
    program, (a, c) = build_single_cursor_copy_program()
    storage = LevelTensorStorage.from_dense(
        [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        (3, 3),
        (0, 1),
        (LevelKind.COMPRESSED, LevelKind.COMPRESSED),
    )
    results = run_program(program, {a: storage}, {c: (3, 3)})
    result = results[c]
    assert result.positions[0] == (0, 2)
    assert result.coordinates[0] == (0, 2)
    assert result.positions[1] == (0, 2, 3)
    assert result.coordinates[1] == (0, 2, 1)
    assert result.values == (1.0, 2.0, 3.0)


# -- adjacent seams stay fail-closed -----------------------------------------


def _seam_cells():
    i, j = IndexVar("i"), IndexVar("j")
    f32 = torch.float32
    posload_operand = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("C", fmt="ss")[i, j],
                CINBinaryOp(
                    Operation.MUL,
                    TensorVar("A", fmt="ss")[i, j],
                    TensorVar("B", fmt="sd")[i, j],
                ),
            ),
        ),
    )
    dense_domains = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("C", fmt="ss")[i, j],
                CINBinaryOp(
                    Operation.MUL,
                    TensorVar("A", fmt="dd")[i, j],
                    TensorVar("B", fmt="dd")[i, j],
                ),
            ),
        ),
    )
    # Recorded seam move: the rank-1 compressed output that used to sit here
    # is the admitted degenerate ordered-stream family
    # (``test_loopir_rank1_assembly_target.py``).  Its neighbour on the same
    # seam -- a single compressed level under two or more dense parents,
    # which the one-dense-prefix rule excludes -- keeps the exact code.
    k = IndexVar("k")
    multi_dense_prefix = ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(
                    TensorVar("C", fmt="dds")[i, j, k],
                    TensorVar("A", fmt="dds")[i, j, k],
                ),
            ),
        ),
    )
    return [
        (
            "posload_co_operand",
            posload_operand,
            (4, 5),
            (((4, 5), f32), ((4, 5), f32)),
            "unsupported_program_shape",
        ),
        (
            "dense_domain_suffix",
            dense_domains,
            (4, 5),
            (((4, 5), f32), ((4, 5), f32)),
            "unsupported_sparse_output",
        ),
        (
            "multi_dense_prefix_sparse_output",
            multi_dense_prefix,
            (3, 4, 5),
            (((3, 4, 5), f32),),
            "unsupported_sparse_output",
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


def test_permuted_mode_order_stays_fail_closed():
    """Nonidentity compressed mode order keeps the recorded boundary."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, c = (builder.new_symbol_id() for _ in range(2))
    decl_a = builder.tensor(
        a,
        "A",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.COMPRESSED, 1),
            builder.level(LevelKind.COMPRESSED, 0),
        ),
    )
    decl_c = builder.tensor(
        c,
        "C",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.COMPRESSED, 0),
            builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    index_j = builder.new_index_id()
    index_i = builder.new_index_id()
    position_a0 = builder.new_position_id()
    position_a1 = builder.new_position_id()
    cursor_a0 = builder.sparse_cursor(
        builder.new_cursor_id(), a, 0, builder.root_position()
    )
    cursor_a1 = builder.sparse_cursor(
        builder.new_cursor_id(), a, 1, builder.position_value(position_a0)
    )
    leaf = builder.append_entry(
        c,
        (builder.index_value(index_i), builder.index_value(index_j)),
        builder.cursor_value(cursor_a1.cursor),
    )
    inner = builder.sparse_for(cursor_a1, position_a1, index_i, builder.block((leaf,)))
    outer = builder.sparse_for(cursor_a0, position_a0, index_j, builder.block((inner,)))
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_c),
        (a,),
        (c,),
        builder.block((outer,)),
    )
    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes={a: (4, 5)},
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=CompilationContext(options),
        )
    assert error.value.defect.code == "unsupported_mode_order"
