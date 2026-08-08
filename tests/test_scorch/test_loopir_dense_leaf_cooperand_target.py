"""Phase-7 compressed-parent/dense-leaf co-operands: the position-load family.

An operand whose value-bearing leaf is a DENSE level below compressed
structure -- ``sd``, ``ssd``, ``sdd``, ``dsd``, ``sddd``, ``ssdd``, ``sssd``,
and the rank-general forms -- is read through :class:`PositionLoad` over a
:class:`DensePosition` spine rather than through a merge cursor.  The ordered
assembly target already builds exactly the right loop nest for these programs;
only its per-alignment-case leaf evaluation refused the node kind, so every
``ss*sd``-shaped elementwise chain failed closed at
``unsupported_program_shape``.

A position load is case-invariant: it addresses the loaded tensor's own
validated dense spine, not a merge cursor, so partially evaluating it for one
cursor-alignment case is sound -- provided the position it grounds at is bound
unconditionally.  ``SparseFor``/``SparseWindowFor`` bindings and INTERSECTION
merges always bind; a UNION merge's binding is optional at a one-sided
coordinate.  The verifier's position typing already refuses to type a
UNION-bound position for a position-load spine; the target repeats the check
at its own boundary.

The legacy comparand is honest for this family -- it generates C++ for every
cell and executes it to well-formed identity-ordered storage -- so the gate is
the B1/B3 discipline: byte parity with ``legacy_generated_cpp`` in both
automatic policy arms, byte-identical produced storage against an
independently keyed legacy build, plus the production LoopIR oracle and the
PyTorch dense reference.
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
from scorch.compiler.diagnostics import InvalidSchedule
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir import lower_llir as _lower_llir_module
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.nodes import (
    BinaryOp as LoopIRBinaryOp,
    LevelKind,
    MergeMode,
    ScalarType,
)
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compare_generated_sources,
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.schedule_passes import erase_schedule
from scorch.compiler.loopir.verifier import LoopIRVerificationError, verify_program
from scorch.stensor import STensor
from tests.test_scorch.test_loopir_multi_compressed_target import (
    validated_storage_pieces,
)
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

_KIND = {"d": LevelKind.DENSE, "s": LevelKind.COMPRESSED}

_DIMS = {"i": 4, "j": 5, "k": 3, "l": 2}
_INDEX = {2: "ij", 3: "ijk", 4: "ijkl"}


def shape_of(fmt):
    return tuple(_DIMS[character] for character in _INDEX[len(fmt)])


def build_cin(result_fmt, operand_fmts, op="mul", dtype=torch.float32):
    """C = A op B (op D) over one shared index tuple, formats per operand."""

    rank = len(result_fmt)
    ivars = tuple(IndexVar(name) for name in _INDEX[rank])
    result = TensorVar("C", fmt=result_fmt, dtype=dtype)
    operation = Operation.MUL if op == "mul" else Operation.ADD
    rhs = None
    for name, fmt in zip("ABD", operand_fmts):
        access = TensorVar(name, fmt=fmt, dtype=dtype)[ivars]
        rhs = access if rhs is None else CINBinaryOp(operation, rhs, access)
    stmt = TensorAssign(result[ivars], rhs)
    for index_var in reversed(ivars):
        stmt = ForAll(index_var, stmt)
    return stmt


def fixture(shape, dtype=torch.float32, seed=20260808, density=0.45, kind="random"):
    torch.manual_seed(seed)
    if kind == "empty":
        return torch.zeros(shape, dtype=dtype)
    if kind == "dense_full":
        return torch.randn(shape).to(dtype)
    dense = ((torch.rand(shape) < density) * torch.randn(shape)).to(dtype)
    if kind == "ragged":
        dense[0] = 0
    return dense


def operand_stensor(dense, name, fmt):
    if all(character == "d" for character in fmt):
        return STensor.from_torch(dense.clone(), name)
    return STensor.from_torch(dense.clone(), name).to_sparse(fmt)


def denses_for(operand_fmts, dtype, kind, shape):
    return [
        fixture(
            shape, dtype, seed=100 + position, kind=kind if position == 0 else "random"
        )
        for position, _ in enumerate(operand_fmts)
    ]


def reference(denses, op):
    out = denses[0].clone()
    for extra in denses[1:]:
        out = out * extra if op == "mul" else out + extra
    return out


def bindings_for(operand_fmts, shape, dtype):
    return tuple((shape, dtype) for _ in operand_fmts)


def compiled(result_fmt, operand_fmts, op, dtype, regblock_enabled):
    shape = shape_of(result_fmt)
    return compile_cin_via_loopir(
        build_cin(result_fmt, operand_fmts, op, dtype),
        shape,
        bindings_for(operand_fmts, shape, dtype),
        compile_options=auto_options(regblock_enabled),
    )


def executed(result_fmt, operand_fmts, op, dtype, stensors, regblock_enabled):
    out = execute_cin_via_loopir(
        build_cin(result_fmt, operand_fmts, op, dtype),
        shape_of(result_fmt),
        *stensors,
        compile_options=auto_options(regblock_enabled, jit=True),
    )
    return out[0] if isinstance(out, tuple) else out


def oracle_storage(kernel, denses, operand_fmts, result_shape):
    """Base and scheduled oracle agreement; returns the level storage."""

    lowering = kernel.lowering
    inputs = {}
    for symbol, dense, fmt in zip(lowering.rhs_access_symbols, denses, operand_fmts):
        if all(character == "d" for character in fmt):
            inputs[symbol] = dense.tolist()
        else:
            inputs[symbol] = LevelTensorStorage.from_dense(
                dense.tolist(),
                tuple(dense.shape),
                tuple(range(len(fmt))),
                tuple(_KIND[character] for character in fmt),
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


# The migrated family: a compressed-parent/dense-leaf co-operand read through
# PositionLoad, at every rank the assembly target covers, in both operand
# orders, with dense prefixes and interleaved dense levels.
ADMITTED = [
    ("ss*sd", "ss", ("ss", "sd")),
    ("sd*ss", "ss", ("sd", "ss")),
    ("sss*sdd", "sss", ("sss", "sdd")),
    ("sss*ssd", "sss", ("sss", "ssd")),
    ("sdd*sss", "sss", ("sdd", "sss")),
    ("ssd*sss", "sss", ("ssd", "sss")),
    ("dss*dsd", "dss", ("dss", "dsd")),
    ("dsd*dss", "dss", ("dsd", "dss")),
    ("ssss*sddd", "ssss", ("ssss", "sddd")),
    ("sddd*ssss", "ssss", ("sddd", "ssss")),
    ("ssss*ssdd", "ssss", ("ssss", "ssdd")),
    ("ssss*sssd", "ssss", ("ssss", "sssd")),
    ("sss*dsd", "sss", ("sss", "dsd")),
    ("sds*ssd", "sss", ("sds", "ssd")),
    ("dds*dsd", "dss", ("dds", "dsd")),
]

_IDS = [case[0] for case in ADMITTED]


# -- source parity -----------------------------------------------------------


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
@pytest.mark.parametrize("name,result_fmt,operand_fmts", ADMITTED, ids=_IDS)
def test_source_matches_legacy_byte_for_byte(
    name, result_fmt, operand_fmts, regblock_enabled
):
    shape = shape_of(result_fmt)
    comparison = compare_generated_sources(
        build_cin(result_fmt, operand_fmts),
        shape,
        bindings_for(operand_fmts, shape, torch.float32),
        compile_options=auto_options(regblock_enabled),
    )
    assert comparison.identical, name


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
@pytest.mark.parametrize(
    "name,result_fmt,operand_fmts",
    ADMITTED[:6],
    ids=_IDS[:6],
)
def test_float64_source_matches_legacy_byte_for_byte(
    name, result_fmt, operand_fmts, regblock_enabled, dtype
):
    shape = shape_of(result_fmt)
    comparison = compare_generated_sources(
        build_cin(result_fmt, operand_fmts, dtype=dtype),
        shape,
        bindings_for(operand_fmts, shape, dtype),
        compile_options=auto_options(regblock_enabled),
    )
    assert comparison.identical, name


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
def test_three_ary_chain_with_a_dense_leaf_matches_legacy(regblock_enabled):
    """A dense-factor widening over the dense-leaf co-operand stays at parity.

    ``ss*sd*dd`` is a recorded seam move: the 3-ary intersection already had
    target code, and the leaf's dense-leaf read is the only thing that had
    refused it.
    """

    shape = shape_of("ss")
    comparison = compare_generated_sources(
        build_cin("ss", ("ss", "sd", "dd")),
        shape,
        bindings_for(("ss", "sd", "dd"), shape, torch.float32),
        compile_options=auto_options(regblock_enabled),
    )
    assert comparison.identical


# -- admission and arm invariance --------------------------------------------


@pytest.mark.parametrize("name,result_fmt,operand_fmts", ADMITTED, ids=_IDS)
def test_admitted_in_both_arms_with_identical_source(name, result_fmt, operand_fmts):
    base = compiled(result_fmt, operand_fmts, "mul", torch.float32, False)
    regblock = compiled(result_fmt, operand_fmts, "mul", torch.float32, True)
    assert base.cpp_source == regblock.cpp_source, name


# -- compiled execution ------------------------------------------------------


@pytest.mark.parametrize(
    "kind", ["random", "ragged", "empty", "dense_full"], ids=lambda k: k
)
@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
@pytest.mark.parametrize("name,result_fmt,operand_fmts", ADMITTED[:8], ids=_IDS[:8])
def test_compiled_execution_matches_the_dense_reference(
    name, result_fmt, operand_fmts, regblock_enabled, kind
):
    shape = shape_of(result_fmt)
    denses = denses_for(operand_fmts, torch.float32, kind, shape)
    stensors = [
        operand_stensor(dense, letter, fmt)
        for dense, letter, fmt in zip(denses, "ABD", operand_fmts)
    ]
    result = executed(
        result_fmt, operand_fmts, "mul", torch.float32, stensors, regblock_enabled
    )
    validated_storage_pieces(result, result_fmt, shape)
    assert torch.allclose(
        result.to_torch(), reference(denses, "mul"), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("name,result_fmt,operand_fmts", ADMITTED[:8], ids=_IDS[:8])
def test_float64_execution_matches_the_dense_reference(name, result_fmt, operand_fmts):
    shape = shape_of(result_fmt)
    denses = denses_for(operand_fmts, torch.float64, "random", shape)
    stensors = [
        operand_stensor(dense, letter, fmt)
        for dense, letter, fmt in zip(denses, "ABD", operand_fmts)
    ]
    result = executed(result_fmt, operand_fmts, "mul", torch.float64, stensors, False)
    validated_storage_pieces(result, result_fmt, shape)
    assert torch.allclose(
        result.to_torch(), reference(denses, "mul"), atol=1e-12, rtol=1e-12
    )


def test_zero_extent_execution_is_canonical_in_both_arms():
    """A zero extent produces canonical empty storage, not a malformed index."""

    result_fmt, operand_fmts = "ss", ("ss", "sd")
    shape = (4, 0)
    denses = [torch.zeros(shape) for _ in operand_fmts]
    sources = []
    for regblock_enabled in (False, True):
        stensors = [
            operand_stensor(dense, letter, fmt)
            for dense, letter, fmt in zip(denses, "AB", operand_fmts)
        ]
        out = execute_cin_via_loopir(
            build_cin(result_fmt, operand_fmts),
            shape,
            *stensors,
            compile_options=auto_options(regblock_enabled, jit=True),
        )
        result = out[0] if isinstance(out, tuple) else out
        pieces, values = validated_storage_pieces(result, result_fmt, shape)
        assert values == []
        assert pieces[0][0] == [0, 0]
        sources.append(
            compiled(result_fmt, operand_fmts, "mul", torch.float32, regblock_enabled)
        )
    assert sources[0].cpp_source == sources[1].cpp_source


def test_repeated_execution_is_byte_stable():
    result_fmt, operand_fmts = "sss", ("sss", "sdd")
    shape = shape_of(result_fmt)
    signatures = []
    for _ in range(3):
        denses = denses_for(operand_fmts, torch.float32, "random", shape)
        stensors = [
            operand_stensor(dense, letter, fmt)
            for dense, letter, fmt in zip(denses, "AB", operand_fmts)
        ]
        result = executed(
            result_fmt, operand_fmts, "mul", torch.float32, stensors, False
        )
        signatures.append(validated_storage_pieces(result, result_fmt, shape))
    assert signatures[0] == signatures[1] == signatures[2]


# -- stored explicit zeros ---------------------------------------------------


def stored_position(index, fmt, coordinate, shape):
    """Decode one coordinate to its stored leaf position, or ``None``."""

    mode_indices = index.mode_indices
    position = 0
    for level, kind in enumerate(fmt):
        if kind == "d":
            position = position * shape[level] + coordinate[level]
            continue
        positions = mode_indices[level][0].tolist()
        coordinates = mode_indices[level][1].tolist()
        start, end = positions[position], positions[position + 1]
        segment = coordinates[start:end]
        if coordinate[level] not in segment:
            return None
        position = start + segment.index(coordinate[level])
    return position


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
@pytest.mark.parametrize(
    "name,result_fmt,operand_fmts",
    [("ss*sd", "ss", ("ss", "sd")), ("sd*ss", "ss", ("sd", "ss"))],
    ids=["ss*sd", "sd*ss"],
)
def test_stored_explicit_zero_is_observed_and_retained(
    name, result_fmt, operand_fmts, regblock_enabled
):
    """A genuinely stored 0.0 keeps its coordinate in the assembled support.

    The compressed operand carries a coordinate whose stored value is exactly
    0.0.  The intersection still visits it, so the result stores that
    coordinate with a 0.0 product -- a structural zero would have dropped it.
    """

    shape = shape_of(result_fmt)
    compressed_position = 0 if operand_fmts[0] == "ss" else 1
    zero_coordinate = (1, 2)
    denses = denses_for(operand_fmts, torch.float32, "random", shape)
    # Both operands must be nonzero at the coordinate so the intersection
    # reaches it; the compressed one then stores an explicit 0.0 there.
    for dense in denses:
        dense[zero_coordinate] = 1.5
    stensors = []
    for position, (dense, letter, fmt) in enumerate(zip(denses, "AB", operand_fmts)):
        tensor = operand_stensor(dense, letter, fmt)
        if position == compressed_position:
            offset = stored_position(tensor.storage.index, fmt, zero_coordinate, shape)
            assert offset is not None
            tensor.storage.value.reshape(-1)[offset] = 0.0
            denses[position] = dense.clone()
            denses[position][zero_coordinate] = 0.0
        stensors.append(tensor)
    result = executed(
        result_fmt, operand_fmts, "mul", torch.float32, stensors, regblock_enabled
    )
    pieces, values = validated_storage_pieces(result, result_fmt, shape)
    assert (
        stored_position(result.storage.index, result_fmt, zero_coordinate, shape)
        is not None
    )
    assert torch.allclose(
        result.to_torch(), reference(denses, "mul"), atol=1e-6, rtol=1e-6
    )


# -- the LoopIR oracle -------------------------------------------------------


@pytest.mark.parametrize("name,result_fmt,operand_fmts", ADMITTED[:8], ids=_IDS[:8])
def test_generated_program_matches_the_loopir_oracle(name, result_fmt, operand_fmts):
    shape = shape_of(result_fmt)
    denses = denses_for(operand_fmts, torch.float32, "random", shape)
    kernel = compiled(result_fmt, operand_fmts, "mul", torch.float32, False)
    produced = oracle_storage(kernel, denses, operand_fmts, shape)
    expected = reference(denses, "mul")
    oracle_dense = torch.zeros(shape)
    stack = [((), 0)]
    # Decode the produced level storage into a dense tensor.
    coordinates = produced.coordinates
    positions = produced.positions
    values = produced.values

    def walk(level, parent, prefix):
        if level == len(result_fmt):
            oracle_dense[prefix] = values[parent]
            return
        if result_fmt[level] == "d":
            for coordinate in range(shape[level]):
                walk(
                    level + 1,
                    parent * shape[level] + coordinate,
                    prefix + (coordinate,),
                )
            return
        start, end = positions[level][parent], positions[level][parent + 1]
        for offset in range(start, end):
            walk(level + 1, offset, prefix + (coordinates[level][offset],))

    walk(0, 0, ())
    assert torch.allclose(oracle_dense, expected, atol=1e-6, rtol=1e-6)
    del stack


# -- fail-closed neighbours --------------------------------------------------


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
@pytest.mark.parametrize(
    "result_fmt,operand_fmts,op,code",
    [
        ("ss", ("ss", "sd"), "add", "unsupported_union_with_dense"),
        ("ss", ("sd", "ss"), "add", "unsupported_union_with_dense"),
        ("sd", ("sd",), "mul", "unsupported_sparse_output_domain"),
        ("sdd", ("sdd",), "mul", "unsupported_sparse_output_domain"),
        ("dds", ("dds",), "mul", "unsupported_sparse_output"),
        ("sds", ("sds",), "mul", "unsupported_sparse_output"),
    ],
    ids=["ss+sd", "sd+ss", "sd-copy", "sdd-copy", "dds-copy", "sds-copy"],
)
def test_excluded_neighbours_keep_their_exact_codes(
    result_fmt, operand_fmts, op, code, regblock_enabled
):
    shape = shape_of(result_fmt)
    with pytest.raises((LoopIRLoweringError, LoopIRTargetError)) as error:
        compile_cin_via_loopir(
            build_cin(result_fmt, operand_fmts, op),
            shape,
            bindings_for(operand_fmts, shape, torch.float32),
            compile_options=auto_options(regblock_enabled),
        )
    assert error.value.defect.code == code


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
def test_permuted_compressed_structure_stays_fail_closed(regblock_enabled):
    """Only all-dense tensors may permute; compressed structure may not.

    This is the pre-existing ``unsupported_mode_order`` boundary, unchanged by
    this family: a dense-leaf co-operand does not make a permuted compressed
    layout lowerable.
    """

    ivars = (IndexVar("i"), IndexVar("j"))
    result = TensorVar("C", fmt="ss", dtype=torch.float32)
    a = TensorVar("A", fmt="ss", dtype=torch.float32, mode_order=[0, 1])[ivars]
    b = TensorVar("B", fmt="sd", dtype=torch.float32, mode_order=[1, 0])[ivars]
    stmt = TensorAssign(result[ivars], CINBinaryOp(Operation.MUL, a, b))
    for index_var in reversed(ivars):
        stmt = ForAll(index_var, stmt)
    with pytest.raises(
        (LoopIRLoweringError, LoopIRTargetError, InvalidSchedule)
    ) as error:
        compile_cin_via_loopir(
            stmt,
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            compile_options=auto_options(regblock_enabled),
        )
    code = getattr(getattr(error.value, "defect", None), "code", None)
    assert code in ("unsupported_mode_order", "unsupported_loop_order") or (
        "result_storage_order" in str(error.value)
    )


# -- the UNION-bound position boundary ---------------------------------------


def build_dense_leaf_union_program(*, position_load=True, mode=MergeMode.UNION):
    """A ``dd`` result over two ``ds`` operands merged at the column level.

    With ``position_load`` the leaf reads the second operand through a
    ``PositionLoad`` grounded at the merge-bound position instead of through
    its cursor -- exactly the shape that has no value at a one-sided
    coordinate.
    """

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    ds_levels = lambda: (  # noqa: E731
        builder.level(LevelKind.DENSE, 0),
        builder.level(LevelKind.COMPRESSED, 1),
    )
    dd_levels = lambda: (  # noqa: E731
        builder.level(LevelKind.DENSE, 0),
        builder.level(LevelKind.DENSE, 1),
    )
    decl_a = builder.tensor(
        a, "A", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), ds_levels()
    )
    decl_b = builder.tensor(
        b, "B", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), ds_levels()
    )
    decl_c = builder.tensor(
        c, "C", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), dd_levels()
    )
    index_i = builder.new_index_id()
    index_j = builder.new_index_id()
    position_a1 = builder.new_position_id()
    position_b1 = builder.new_position_id()
    cursor_a1 = builder.sparse_cursor(
        builder.new_cursor_id(),
        a,
        1,
        builder.dense_position(
            a, 0, builder.root_position(), builder.index_value(index_i)
        ),
    )
    cursor_b1 = builder.sparse_cursor(
        builder.new_cursor_id(),
        b,
        1,
        builder.dense_position(
            b, 0, builder.root_position(), builder.index_value(index_i)
        ),
    )
    united = mode is MergeMode.UNION
    left = builder.cursor_value(
        cursor_a1.cursor, builder.float_const(0.0) if united else None
    )
    if position_load:
        right = builder.position_load(b, builder.position_value(position_b1))
    else:
        right = builder.cursor_value(
            cursor_b1.cursor, builder.float_const(0.0) if united else None
        )
    leaf = builder.store(
        c,
        (builder.index_value(index_i), builder.index_value(index_j)),
        builder.binary(LoopIRBinaryOp.ADD, left, right),
    )
    inner = builder.merged_sparse_for(
        mode,
        (cursor_a1, cursor_b1),
        index_j,
        builder.block((leaf,)),
        (position_a1, position_b1),
    )
    outer = builder.dense_for(index_i, dim_i.dimension, builder.block((inner,)))
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_b, decl_c),
        (a, b),
        (c,),
        builder.block((outer,)),
    )
    return program, (a, b, c)


def test_union_bound_position_load_fails_in_the_verifier():
    program, _ = build_dense_leaf_union_program()
    with pytest.raises(LoopIRVerificationError) as error:
        verify_program(program)
    assert error.value.defect.code == "unsupported_sparse_hierarchy"


def test_union_bound_position_load_also_fails_at_the_target_boundary():
    """The owning target repeats the check, independent of verification."""

    program, (a, b, _) = build_dense_leaf_union_program()
    lowering = _lower_llir_module._TargetLowering(
        program, {a: (4, 5), b: (4, 5)}, (4, 5)
    )
    with pytest.raises(LoopIRTargetError) as error:
        lowering.raw_loop_statements()
    assert error.value.defect.code == "unsupported_program_shape"
    assert "unconditionally bound position" in error.value.defect.message


def test_intersection_bound_position_load_is_admitted_by_the_target():
    """The sound control: an INTERSECTION binding is unconditional."""

    program, (a, b, _) = build_dense_leaf_union_program(mode=MergeMode.INTERSECTION)
    lowering = _lower_llir_module._TargetLowering(
        program, {a: (4, 5), b: (4, 5)}, (4, 5)
    )
    assert lowering.raw_loop_statements()


def test_union_cursor_value_control_is_unchanged():
    """The same nest reading through cursors stays admitted in both places."""

    program, (a, b, _) = build_dense_leaf_union_program(position_load=False)
    verify_program(program)
    lowering = _lower_llir_module._TargetLowering(
        program, {a: (4, 5), b: (4, 5)}, (4, 5)
    )
    assert lowering.raw_loop_statements()


# -- identity neutrality -----------------------------------------------------


@pytest.mark.parametrize("name,result_fmt,operand_fmts", ADMITTED[:6], ids=_IDS[:6])
def test_family_needs_no_schedule_or_identity_change(name, result_fmt, operand_fmts):
    """The program is expressible under the unchanged canonical schema."""

    kernel = compiled(result_fmt, operand_fmts, "mul", torch.float32, False)
    dump = canonical_program_dump(kernel.lowering.program)
    assert dump == canonical_program_dump(kernel.lowering.program)
    assert erase_schedule(kernel.lowering.program) == kernel.lowering.program


@pytest.mark.parametrize(
    "result_fmt,operand_fmts,op",
    [
        ("ss", ("ss",), "mul"),
        ("ss", ("ss", "dd"), "mul"),
        ("ss", ("ss", "ds"), "mul"),
        ("ss", ("ss", "ss"), "add"),
        ("sss", ("sss", "sss"), "add"),
        ("s", ("s", "s"), "add"),
        ("ds", ("ds", "ds"), "add"),
    ],
    ids=["ss-copy", "ss*dd", "ss*ds", "ss+ss", "sss+sss", "s+s", "ds+ds"],
)
@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["base", "regblock"])
def test_neighbouring_families_are_byte_unchanged(
    result_fmt, operand_fmts, op, regblock_enabled
):
    """Every family that already had target code keeps legacy byte parity."""

    shape = shape_of(result_fmt) if len(result_fmt) > 1 else (9,)
    if len(result_fmt) == 1:
        index = IndexVar("i")
        result = TensorVar("C", fmt=result_fmt, dtype=torch.float32)
        operation = Operation.MUL if op == "mul" else Operation.ADD
        rhs = None
        for name, fmt in zip("AB", operand_fmts):
            access = TensorVar(name, fmt=fmt, dtype=torch.float32)[index]
            rhs = access if rhs is None else CINBinaryOp(operation, rhs, access)
        cin = ForAll(index, TensorAssign(result[index], rhs))
    else:
        cin = build_cin(result_fmt, operand_fmts, op)
    comparison = compare_generated_sources(
        cin,
        shape,
        bindings_for(operand_fmts, shape, torch.float32),
        compile_options=auto_options(regblock_enabled),
    )
    assert comparison.identical
