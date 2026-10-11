"""Phase-7 mixed dense-leaf operand loads: the ``sd``-family load chain.

Operand tensors whose compressed structure sits above a dense
value-bearing sub-tree (``sd``, ``sdd``, ``dsd``) lower through declared
:class:`PositionLoad` reads — a dense-position spine grounded at the
single-cursor bound row position — into the existing dense-result
families.  Rank-1 ``s`` is the adjacent compressed-leaf control and keeps
the established :class:`CursorValue` representation.  The generated C++
must be byte-identical to the legacy pipeline's long-proven dense-output
kernels in both automatic policy arms, execute through the shared JIT
build path, and match the production LoopIR oracle and the PyTorch dense
reference.

Runtime ``sd``-family inputs are built through the public
``to_sparse`` dense-suffix materialization (the previously recorded
runtime gap, now closed); the hand-built ``TensorIndex`` builders remain
as comparands whose exact-storage equivalence is locked in
``test_to_sparse_dense_suffix.py``.
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
from scorch.compiler.identity import SymbolId
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
    legacy_generated_cpp,
)
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.schedule_passes import erase_schedule
from scorch.stensor import STensor
from scorch.storage import TensorIndex
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

_KIND = {
    "d": LevelKind.DENSE,
    "s": LevelKind.COMPRESSED,
}


# -- CIN builders ------------------------------------------------------------


def build_copy_cin(dtype=torch.float32, a_fmt="sd"):
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt=a_fmt, dtype=dtype)
    c = TensorVar("C", fmt="dd", dtype=dtype)
    return ForAll(i, ForAll(j, TensorAssign(c[i, j], a[i, j])))


def build_mul_cin(dtype=torch.float32, *, commuted=False):
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="sd", dtype=dtype)
    b = TensorVar("B", fmt="dd", dtype=dtype)
    c = TensorVar("C", fmt="dd", dtype=dtype)
    value = (
        CINBinaryOp(Operation.MUL, b[i, j], a[i, j])
        if commuted
        else CINBinaryOp(Operation.MUL, a[i, j], b[i, j])
    )
    return ForAll(i, ForAll(j, TensorAssign(c[i, j], value)))


def build_spmv_cin(dtype=torch.float32):
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="sd", dtype=dtype)
    x = TensorVar("x", fmt="d", dtype=dtype)
    y = TensorVar("y", fmt="d", dtype=dtype)
    assign = TensorAssign(
        y[i], CINBinaryOp(Operation.MUL, a[i, j], x[j]), op=Operation.ADD
    )
    return ForAll(i, ForAll(j, assign))


def build_reduce_cin(dtype=torch.float32):
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="sd", dtype=dtype)
    y = TensorVar("y", fmt="d", dtype=dtype)
    return ForAll(i, ForAll(j, TensorAssign(y[i], a[i, j], op=Operation.ADD)))


def build_matmul_cin(dtype=torch.float32):
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="sd", dtype=dtype)
    b = TensorVar("B", fmt="dd", dtype=dtype)
    c = TensorVar("C", fmt="dd", dtype=dtype)
    assign = TensorAssign(
        c[i, k],
        CINBinaryOp(Operation.MUL, a[i, j], b[j, k]),
        op=Operation.ADD,
    )
    return ForAll(i, ForAll(j, ForAll(k, assign)))


def build_rank3_copy_cin(a_fmt, dtype=torch.float32):
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt=a_fmt, dtype=dtype)
    c = TensorVar("C", fmt="ddd", dtype=dtype)
    return ForAll(i, ForAll(j, ForAll(k, TensorAssign(c[i, j, k], a[i, j, k]))))


def build_rank1_copy_cin(dtype=torch.float32):
    i = IndexVar("i")
    a = TensorVar("A", fmt="s", dtype=dtype)
    c = TensorVar("C", fmt="d", dtype=dtype)
    return ForAll(i, TensorAssign(c[i], a[i]))


# -- runtime storage builders ------------------------------------------------


def sparse_sd(dense, name):
    """Compressed stored rows over dense row blocks (fmt ``sd``).

    Kept as the hand-built comparand:
    ``test_to_sparse_dense_suffix.py`` proves the public conversion
    produces exactly this storage, and the runtime batteries now build
    their inputs through ``to_sparse``.
    """

    rows = [r for r in range(dense.shape[0]) if bool(dense[r].ne(0).any())]
    pos = torch.tensor([0, len(rows)], dtype=torch.int32)
    crd = torch.tensor(rows, dtype=torch.int32)
    values = (
        torch.stack([dense[r] for r in rows]).reshape(-1).clone()
        if rows
        else dense.new_zeros((0,))
    )
    return STensor(
        name=name,
        shape=tuple(dense.shape),
        index=TensorIndex("sd", [[pos, crd], []]),
        value=values,
    )


def sparse_sdd(dense, name):
    rows = [r for r in range(dense.shape[0]) if bool(dense[r].ne(0).any())]
    pos = torch.tensor([0, len(rows)], dtype=torch.int32)
    crd = torch.tensor(rows, dtype=torch.int32)
    values = (
        torch.stack([dense[r] for r in rows]).reshape(-1).clone()
        if rows
        else dense.new_zeros((0,))
    )
    return STensor(
        name=name,
        shape=tuple(dense.shape),
        index=TensorIndex("sdd", [[pos, crd], [], []]),
        value=values,
    )


def sparse_dsd(dense, name):
    d0, d1, _ = dense.shape
    pos = [0]
    crd = []
    blocks = []
    for i in range(d0):
        stored = [j for j in range(d1) if bool(dense[i, j].ne(0).any())]
        crd.extend(stored)
        pos.append(len(crd))
        for j in stored:
            blocks.append(dense[i, j])
    values = (
        torch.stack(blocks).reshape(-1).clone() if blocks else dense.new_zeros((0,))
    )
    return STensor(
        name=name,
        shape=tuple(dense.shape),
        index=TensorIndex(
            "dsd",
            [
                [],
                [
                    torch.tensor(pos, dtype=torch.int32),
                    torch.tensor(crd, dtype=torch.int32),
                ],
                [],
            ],
        ),
        value=values,
    )


def converted_sd(dense, name):
    """Public-conversion ``sd`` runtime input (the closed runtime gap)."""

    return STensor.from_torch(dense.clone(), name).to_sparse("sd")


def converted_sdd(dense, name):
    return STensor.from_torch(dense.clone(), name).to_sparse("sdd")


def converted_dsd(dense, name):
    return STensor.from_torch(dense.clone(), name).to_sparse("dsd")


def sparse_s(dense, name):
    idxs = [i for i in range(dense.shape[0]) if bool(dense[i].ne(0))]
    pos = torch.tensor([0, len(idxs)], dtype=torch.int32)
    crd = torch.tensor(idxs, dtype=torch.int32)
    values = dense[idxs].clone() if idxs else dense.new_zeros((0,))
    return STensor(
        name=name,
        shape=tuple(dense.shape),
        index=TensorIndex("s", [[pos, crd]]),
        value=values,
    )


def dense_stensor(tensor, name):
    return STensor.from_torch(tensor.clone(), name).to_dense()


def mixed_storage(dense, fmt):
    """Bind one logical dense tensor as oracle-level mixed storage."""

    return LevelTensorStorage.from_dense(
        dense.tolist(),
        tuple(dense.shape),
        tuple(range(len(fmt))),
        tuple(_KIND[ch] for ch in fmt),
    )


def oracle_dense_result(kernel, inputs_by_fmt, result_shape):
    """Base and scheduled oracle agreement; returns the dense result."""

    lowering = kernel.lowering
    inputs = {}
    for symbol, (dense, fmt) in zip(lowering.rhs_access_symbols, inputs_by_fmt):
        if all(ch == "d" for ch in fmt):
            inputs[symbol] = dense.tolist()
        else:
            inputs[symbol] = mixed_storage(dense, fmt)
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


def executed_dense(cin, result_shape, stensors, regblock_enabled):
    out = execute_cin_via_loopir(
        cin,
        result_shape,
        *stensors,
        compile_options=auto_options(regblock_enabled, jit=True),
    )
    result = out[0] if isinstance(out, tuple) else out
    dense = result.to_torch() if isinstance(result, STensor) else result
    return torch.as_tensor(dense).reshape(result_shape)


# -- source parity -----------------------------------------------------------

_PARITY_CELLS = [
    ("copy", build_copy_cin, (4, 5), (("sd", (4, 5)),)),
    ("mul", build_mul_cin, (4, 5), (("sd", (4, 5)), ("dd", (4, 5)))),
    (
        "mul_commuted",
        lambda dtype=torch.float32: build_mul_cin(dtype, commuted=True),
        (4, 5),
        (("dd", (4, 5)), ("sd", (4, 5))),
    ),
    ("spmv", build_spmv_cin, (4,), (("sd", (4, 6)), ("d", (6,)))),
    ("reduce", build_reduce_cin, (4,), (("sd", (4, 5)),)),
    ("matmul", build_matmul_cin, (4, 5), (("sd", (4, 6)), ("dd", (6, 5)))),
    (
        "rank3_sdd",
        lambda dtype=torch.float32: build_rank3_copy_cin("sdd", dtype),
        (3, 4, 5),
        (("sdd", (3, 4, 5)),),
    ),
    (
        "rank3_dsd",
        lambda dtype=torch.float32: build_rank3_copy_cin("dsd", dtype),
        (3, 4, 5),
        (("dsd", (3, 4, 5)),),
    ),
    ("rank1_copy", build_rank1_copy_cin, (4,), (("s", (4,)),)),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("cell", _PARITY_CELLS, ids=[cell[0] for cell in _PARITY_CELLS])
def test_source_parity_matches_legacy_in_both_arms(cell, dtype):
    _, build, result_shape, operands = cell
    bindings = tuple((shape, dtype) for _, shape in operands)
    for arm in (False, True):
        comparison = compare_generated_sources(
            build(dtype),
            result_shape,
            bindings,
            compile_options=auto_options(arm),
        )
        assert comparison.identical, f"arm regblock={arm} diverged"


def test_regblock_arms_genuinely_diverge_on_the_dense_contraction():
    """The matmul cell proves per-arm parity is a real gate, not vacuous."""

    sources = {}
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            build_matmul_cin(),
            (4, 5),
            (((4, 6), torch.float32), ((6, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        sources[arm] = kernel.cpp_source
    assert sources[False] != sources[True]


def test_generated_source_contains_the_physical_load_chain():
    kernel = compile_cin_via_loopir(
        build_copy_cin(),
        (4, 5),
        (((4, 5), torch.float32),),
        compile_options=auto_options(False),
    )
    assert "int pA0_end = A0_pos[1];" in kernel.cpp_source
    assert "for (int pA0 = A0_pos[0]; pA0 < pA0_end; pA0++)" in kernel.cpp_source
    assert "position_load" in kernel.program_dump


# -- compiled execution ------------------------------------------------------


def _sd_fixture(dtype=torch.float32):
    dense_a = torch.tensor(
        [
            [1.0, 0.0, 2.0, 0.0, 0.0, -3.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 4.5, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    dense_b = torch.arange(30, dtype=dtype).reshape(6, 5) / 7.0
    return dense_a, dense_b


@pytest.mark.parametrize("regblock_enabled", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_matmul_matches_every_reference(regblock_enabled, dtype):
    dense_a, dense_b = _sd_fixture(dtype)
    cin = build_matmul_cin(dtype)
    got = executed_dense(
        cin,
        (4, 5),
        (converted_sd(dense_a, "A"), dense_stensor(dense_b, "B")),
        regblock_enabled,
    )
    expected = dense_a @ dense_b
    assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3)

    kernel = compile_cin_via_loopir(
        cin,
        (4, 5),
        (((4, 6), dtype), ((6, 5), dtype)),
        compile_options=auto_options(regblock_enabled),
    )
    oracle = oracle_dense_result(kernel, ((dense_a, "sd"), (dense_b, "dd")), (4, 5))
    oracle_tensor = torch.tensor(oracle, dtype=dtype)
    assert torch.allclose(got, oracle_tensor, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_compiled_copy_preserves_empty_and_explicit_zero_rows(regblock_enabled):
    dense = torch.zeros(5, 3)
    dense[0, 1] = 2.0
    dense[3, 0] = -1.0
    st = converted_sd(dense, "A")
    got = executed_dense(build_copy_cin(), (5, 3), (st,), regblock_enabled)
    assert torch.equal(got, dense)

    # A stored row whose dense block carries an explicit zero still copies
    # that block exactly (the load is physical, not pruned).
    explicit = converted_sd(dense, "A")
    got = executed_dense(build_copy_cin(), (5, 3), (explicit,), regblock_enabled)
    assert torch.equal(got, dense)


@pytest.mark.parametrize("fmt", ["sdd", "dsd"])
def test_compiled_rank3_copy_matches_reference(fmt):
    dense = torch.zeros(3, 4, 5)
    dense[0, 1] = torch.arange(5, dtype=torch.float32)
    dense[2, 0, 3] = 7.0
    dense[2, 3, 4] = -2.5
    builder = converted_sdd if fmt == "sdd" else converted_dsd
    got = executed_dense(
        build_rank3_copy_cin(fmt), (3, 4, 5), (builder(dense, "A"),), False
    )
    assert torch.equal(got, dense)


def test_compiled_spmv_and_reduce_match_reference():
    dense_a, _ = _sd_fixture()
    x = torch.arange(6, dtype=torch.float32) / 3.0
    got = executed_dense(
        build_spmv_cin(),
        (4,),
        (converted_sd(dense_a, "A"), dense_stensor(x, "x")),
        False,
    )
    assert torch.allclose(got, dense_a @ x, atol=1e-3, rtol=1e-3)

    got = executed_dense(build_reduce_cin(), (4,), (converted_sd(dense_a, "A"),), False)
    assert torch.allclose(got, dense_a.sum(dim=1), atol=1e-3, rtol=1e-3)


def test_compiled_rank1_copy_matches_reference():
    vector = torch.tensor([0.0, 2.0, 0.0, -1.5])
    got = executed_dense(build_rank1_copy_cin(), (4,), (sparse_s(vector, "A"),), False)
    assert torch.equal(got, vector)


@pytest.mark.parametrize("a_shape", [(0, 5), (4, 0)], ids=["zero_rows", "zero_cols"])
def test_zero_extent_cells(a_shape):
    dense = torch.zeros(a_shape)
    got = executed_dense(build_copy_cin(), a_shape, (converted_sd(dense, "A"),), False)
    assert torch.equal(got, dense)


def test_ragged_supports_and_repeated_public_execution():
    torch.manual_seed(20260730)
    dense_a = torch.randn(6, 7) * (torch.rand(6, 7) < 0.35)
    dense_b = torch.randn(7, 3)
    expected = dense_a @ dense_b
    for _ in range(5):
        got = executed_dense(
            build_matmul_cin(),
            (6, 3),
            (converted_sd(dense_a, "A"), dense_stensor(dense_b, "B")),
            False,
        )
        assert torch.allclose(got, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_execution_matches_independent_legacy_build(regblock_enabled):
    """The independent marker-keyed legacy module returns the same result."""

    from tests.test_scorch.test_loopir_sparse_workspace_target import (
        execute_legacy_module,
    )

    dense_a, dense_b = _sd_fixture()
    options = auto_options(regblock_enabled, jit=True)
    cpp = legacy_generated_cpp(
        build_matmul_cin(),
        (4, 5),
        (((4, 6), torch.float32), ((6, 5), torch.float32)),
        compile_options=options,
    )
    legacy = execute_legacy_module(
        cpp,
        (4, 5),
        converted_sd(dense_a, "A"),
        dense_stensor(dense_b, "B"),
        options,
    )
    legacy_dense = torch.as_tensor(legacy.storage.value).reshape(4, 5)
    got = executed_dense(
        build_matmul_cin(),
        (4, 5),
        (converted_sd(dense_a, "A"), dense_stensor(dense_b, "B")),
        regblock_enabled,
    )
    assert torch.allclose(got, legacy_dense, atol=1e-5, rtol=1e-5)


# -- route ownership, dumps, and erasure -------------------------------------


def test_canonical_dump_is_arm_stable_and_erases_to_base():
    dumps = []
    for arm in (False, True):
        kernel = compile_cin_via_loopir(
            build_copy_cin(),
            (4, 5),
            (((4, 5), torch.float32),),
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
    assert '"schema":"scorch.loopir.canonical.v13"' in dumps[0]
    assert '"kind":"position_load"' in dumps[0]


def test_rank1_compressed_leaf_keeps_the_cursor_value_representation():
    """PositionLoad originates only for a dense leaf below sparse structure."""

    kernel = compile_cin_via_loopir(
        build_rank1_copy_cin(),
        (4,),
        (((4,), torch.float32),),
        compile_options=auto_options(False),
    )
    assert '"kind":"cursor_value"' in kernel.program_dump
    assert '"kind":"position_load"' not in kernel.program_dump


def test_compile_is_deterministic_per_arm():
    for arm in (False, True):
        first = compile_cin_via_loopir(
            build_mul_cin(),
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        second = compile_cin_via_loopir(
            build_mul_cin(),
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            compile_options=auto_options(arm),
        )
        assert first.cpp_source == second.cpp_source
        assert first.request_identity == second.request_identity


# -- hand-built target programs ----------------------------------------------


def build_mixed_copy_program(*, forge=None):
    """The exact sd-copy LoopIR program the pipeline lowers, hand-built."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    sym_a, sym_c = (builder.new_symbol_id() for _ in range(2))
    decl_a = builder.tensor(
        sym_a,
        "A",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.COMPRESSED, 0),
            builder.level(LevelKind.DENSE, 1),
        ),
    )
    decl_c = builder.tensor(
        sym_c,
        "C",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        builder.dense_levels(2),
    )
    index_i = builder.new_index_id()
    index_j = builder.new_index_id()
    position = builder.new_position_id()
    cursor = builder.sparse_cursor(
        builder.new_cursor_id(), sym_a, 0, builder.root_position()
    )
    if forge is None:
        load_position = builder.dense_position(
            sym_a, 1, builder.position_value(position), builder.index_value(index_j)
        )
    else:
        load_position = forge(builder, sym_a, position, index_i, index_j)
    leaf = builder.store(
        sym_c,
        (builder.index_value(index_i), builder.index_value(index_j)),
        builder.position_load(sym_a, load_position),
    )
    inner = builder.dense_for(index_j, dim_j.dimension, builder.block((leaf,)))
    outer = builder.sparse_for(cursor, position, index_i, builder.block((inner,)))
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_c),
        (sym_a,),
        (sym_c,),
        builder.block((outer,)),
    )
    return program, sym_a, sym_c


def test_hand_built_program_lowers_and_matches_pipeline_source():
    program, sym_a, _ = build_mixed_copy_program()
    options = CompileOptions.from_environment(environ={})
    context = CompilationContext(options)
    function = lower_loopir_to_llir(
        program,
        input_shapes={sym_a: (4, 5)},
        result_shape=(4, 5),
        compile_options=options,
        compilation_context=context,
    )
    from scorch.ops import _lower_generated_llir

    cpp_source = _lower_generated_llir(function, options, context)
    kernel = compile_cin_via_loopir(
        build_copy_cin(),
        (4, 5),
        (((4, 5), torch.float32),),
        compile_options=auto_options(False),
    )
    assert cpp_source == kernel.cpp_source


def test_forged_spine_coordinate_fails_closed_at_the_verifier():
    """A spine coordinate from the wrong dimension never reaches the target."""

    from scorch.compiler.loopir.verifier import LoopIRVerificationError

    def forge(builder, sym_a, position, index_i, index_j):
        return builder.dense_position(
            sym_a, 1, builder.position_value(position), builder.index_value(index_i)
        )

    program, sym_a, _ = build_mixed_copy_program(forge=forge)
    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRVerificationError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes={sym_a: (4, 5)},
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=CompilationContext(options),
        )
    assert error.value.defect.code == "domain_mismatch"


@pytest.mark.parametrize("callback", ["iter", "getitem"])
def test_input_shape_mapping_cannot_mutate_position_load_after_verification(
    callback,
):
    """Caller mapping callbacks run before the target trusts the program."""

    from scorch.compiler.loopir.verifier import LoopIRVerificationError

    program, sym_a, _ = build_mixed_copy_program()
    load = program.body.statements[0].body.statements[0].body.statements[0].value

    class MutatingShapes(dict):
        def __iter__(self):
            if callback == "iter":
                object.__setattr__(load, "tensor", [])
            return super().__iter__()

        def __getitem__(self, key):
            if callback == "getitem":
                object.__setattr__(load, "tensor", [])
            return super().__getitem__(key)

    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRVerificationError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes=MutatingShapes({sym_a: (4, 5)}),
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=CompilationContext(options),
        )
    assert error.value.defect.code == "invalid_symbol_id"


def test_malformed_position_load_fails_before_input_shape_mapping_callbacks():
    """The initial verifier remains cheaper than a caller-controlled mapping."""

    from scorch.compiler.loopir.verifier import LoopIRVerificationError

    program, sym_a, _ = build_mixed_copy_program()
    load = program.body.statements[0].body.statements[0].body.statements[0].value
    object.__setattr__(load, "tensor", [])
    called = False

    class HostileShapes(dict):
        def __iter__(self):
            nonlocal called
            called = True
            raise RuntimeError("shape mapping must not run")

    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRVerificationError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes=HostileShapes({sym_a: (4, 5)}),
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=CompilationContext(options),
        )
    assert error.value.defect.code == "invalid_symbol_id"
    assert not called


def test_hostile_shape_value_object_is_rejected_without_invocation():
    """A tuple-subclass shape value fails the exact-type check uninvoked."""

    program, sym_a, _ = build_mixed_copy_program()
    load = program.body.statements[0].body.statements[0].body.statements[0].value
    invoked = []

    class HostileShape(tuple):
        def __len__(self):
            invoked.append("len")
            object.__setattr__(load, "tensor", [])
            return tuple.__len__(self)

        def __iter__(self):
            invoked.append("iter")
            object.__setattr__(load, "tensor", [])
            return tuple.__iter__(self)

        def __eq__(self, other):
            invoked.append("eq")
            return tuple.__eq__(self, other)

        __hash__ = tuple.__hash__

    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes={sym_a: HostileShape((4, 5))},
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=CompilationContext(options),
        )
    assert error.value.defect.code == "invalid_shape_binding"
    assert invoked == []


def test_duplicate_shape_mapping_keys_fail_closed():
    """A Mapping iterating one SymbolId twice cannot pass the snapshot."""

    program, sym_a, _ = build_mixed_copy_program()

    class DuplicateKeys(dict):
        def __iter__(self):
            yield SymbolId(sym_a.value)
            yield SymbolId(sym_a.value)

    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes=DuplicateKeys({sym_a: (4, 5)}),
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=CompilationContext(options),
        )
    assert error.value.defect.code == "invalid_shape_binding"
    assert "unique" in error.value.defect.message


def test_shape_mapping_key_mutation_during_lookup_fails_closed():
    """Mutating a SymbolId key inside ``__getitem__`` cannot pass snapshot."""

    program, sym_a, _ = build_mixed_copy_program()

    class KeyMutating(dict):
        def __getitem__(self, key):
            value = super().__getitem__(key)
            object.__setattr__(key, "value", key.value + 1000)
            return value

    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes=KeyMutating({sym_a: (4, 5)}),
            result_shape=(4, 5),
            compile_options=options,
            compilation_context=CompilationContext(options),
        )
    assert error.value.defect.code == "invalid_shape_binding"
    assert "changed" in error.value.defect.message


def build_hierarchical_mixed_program():
    """A verified ssd-input copy: two compressed levels above a dense leaf."""

    builder = LoopIRBuilder()
    dims = tuple(builder.dimension(name) for name in ("i", "j", "k"))
    sym_a, sym_c = (builder.new_symbol_id() for _ in range(2))
    decl_a = builder.tensor(
        sym_a,
        "A",
        ScalarType.FLOAT32,
        tuple(dim.dimension for dim in dims),
        (
            builder.level(LevelKind.COMPRESSED, 0),
            builder.level(LevelKind.COMPRESSED, 1),
            builder.level(LevelKind.DENSE, 2),
        ),
    )
    decl_c = builder.tensor(
        sym_c,
        "C",
        ScalarType.FLOAT32,
        tuple(dim.dimension for dim in dims),
        builder.dense_levels(3),
    )
    indices = tuple(builder.new_index_id() for _ in range(3))
    position_0 = builder.new_position_id()
    position_1 = builder.new_position_id()
    cursor_0 = builder.sparse_cursor(
        builder.new_cursor_id(), sym_a, 0, builder.root_position()
    )
    cursor_1 = builder.sparse_cursor(
        builder.new_cursor_id(), sym_a, 1, builder.position_value(position_0)
    )
    leaf = builder.store(
        sym_c,
        tuple(builder.index_value(index) for index in indices),
        builder.position_load(
            sym_a,
            builder.dense_position(
                sym_a,
                2,
                builder.position_value(position_1),
                builder.index_value(indices[2]),
            ),
        ),
    )
    inner = builder.dense_for(indices[2], dims[2].dimension, builder.block((leaf,)))
    middle = builder.sparse_for(
        cursor_1, position_1, indices[1], builder.block((inner,))
    )
    outer = builder.sparse_for(
        cursor_0, position_0, indices[0], builder.block((middle,))
    )
    return (
        builder.program(
            dims, (decl_a, decl_c), (sym_a,), (sym_c,), builder.block((outer,))
        ),
        sym_a,
    )


def test_hierarchical_descent_fails_closed_at_the_target():
    """The verifier admits the ssd descent; the target owns the rejection."""

    program, sym_a = build_hierarchical_mixed_program()
    options = CompileOptions.from_environment(environ={})
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program,
            input_shapes={sym_a: (3, 4, 5)},
            result_shape=(3, 4, 5),
            compile_options=options,
            compilation_context=CompilationContext(options),
        )
    assert error.value.defect.code == "unsupported_program_shape"


def test_target_failure_is_a_recorded_stage_loss():
    program, sym_a = build_hierarchical_mixed_program()
    options = CompileOptions.from_environment(environ={})
    context = CompilationContext(options)
    with pytest.raises(LoopIRTargetError):
        lower_loopir_to_llir(
            program,
            input_shapes={sym_a: (3, 4, 5)},
            result_shape=(3, 4, 5),
            compile_options=options,
            compilation_context=context,
        )
    assert context._failed_stage_id is CompilerStageId.LOOPIR_TO_LLIR_LOWERING
    assert CompilerStageId.LOOPIR_TO_LLIR_LOWERING not in {
        record.stage_id for record in context.stage_run_records
    }


# -- adjacent seams stay fail-closed -----------------------------------------


def _seam_cells():
    i, j = IndexVar("i"), IndexVar("j")
    i3, j3, k3 = IndexVar("i"), IndexVar("j"), IndexVar("k")

    merged = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("C", fmt="dd")[i, j],
                CINBinaryOp(
                    Operation.MUL,
                    TensorVar("A", fmt="sd")[i, j],
                    TensorVar("B", fmt="sd")[i, j],
                ),
            ),
        ),
    )
    union = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("C", fmt="dd")[i, j],
                CINBinaryOp(
                    Operation.ADD,
                    TensorVar("A", fmt="sd")[i, j],
                    TensorVar("B", fmt="sd")[i, j],
                ),
            ),
        ),
    )
    hierarchical = ForAll(
        i3,
        ForAll(
            j3,
            ForAll(
                k3,
                TensorAssign(
                    TensorVar("C", fmt="ddd")[i3, j3, k3],
                    TensorVar("A", fmt="ssd")[i3, j3, k3],
                ),
            ),
        ),
    )
    sparse_result = ForAll(
        i,
        ForAll(
            j,
            TensorAssign(
                TensorVar("C", fmt="sd")[i, j], TensorVar("A", fmt="sd")[i, j]
            ),
        ),
    )
    permuted = ForAll(
        j,
        ForAll(
            i,
            TensorAssign(
                TensorVar("C", fmt="dd", mode_order=(1, 0))[i, j],
                TensorVar("A", fmt="sd", mode_order=(1, 0))[i, j],
            ),
        ),
    )
    i1 = IndexVar("i")
    merged_outermost = ForAll(
        i1,
        TensorAssign(
            TensorVar("C", fmt="d")[i1],
            CINBinaryOp(
                Operation.MUL,
                TensorVar("A", fmt="s")[i1],
                TensorVar("B", fmt="s")[i1],
            ),
        ),
    )
    return [
        (
            "merged_mixed_intersection",
            merged,
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            "unsupported_sparse_hierarchy",
        ),
        (
            "merged_mixed_union",
            union,
            (4, 5),
            (((4, 5), torch.float32), ((4, 5), torch.float32)),
            "unsupported_sparse_hierarchy",
        ),
        (
            "hierarchical_compressed_above_dense_leaf",
            hierarchical,
            (3, 4, 5),
            (((3, 4, 5), torch.float32),),
            "unsupported_program_shape",
        ),
        (
            "mixed_operand_into_sparse_domain_result",
            sparse_result,
            (4, 5),
            (((4, 5), torch.float32),),
            "unsupported_sparse_output_domain",
        ),
        (
            "permuted_compressed_pair",
            permuted,
            (4, 5),
            (((4, 5), torch.float32),),
            "unsupported_mode_order",
        ),
        (
            "merged_outermost_vectors",
            merged_outermost,
            (4,),
            (((4,), torch.float32), ((4,), torch.float32)),
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


# -- thread-count invariance ---------------------------------------------------

_THREAD_PROBE = """
import json
import torch
from tests.test_scorch.test_loopir_mixed_operand_target import (
    build_matmul_cin,
    converted_sd,
    dense_stensor,
    executed_dense,
)

torch.manual_seed(20260730)
dense_a = (torch.rand(9, 8) < 0.4) * torch.randn(9, 8)
dense_b = torch.randn(8, 6)
got = executed_dense(
    build_matmul_cin(),
    (9, 6),
    (converted_sd(dense_a, "A"), dense_stensor(dense_b, "B")),
    False,
)
print(json.dumps({"values": got.reshape(-1).tolist()}))
"""


def test_thread_count_invariance_and_race_freedom():
    """Distinct OpenMP pools produce identical dense results.

    The outer position loop parallelizes over stored rows; every thread
    writes disjoint dense result rows, so two fresh processes under
    different ``OMP_NUM_THREADS`` must agree exactly.  The parent process
    compiles first so both children share the session's isolated JIT
    cache.
    """

    import json
    import os
    import subprocess
    import sys

    torch.manual_seed(20260730)
    dense_a = (torch.rand(9, 8) < 0.4) * torch.randn(9, 8)
    dense_b = torch.randn(8, 6)
    executed_dense(
        build_matmul_cin(),
        (9, 6),
        (converted_sd(dense_a, "A"), dense_stensor(dense_b, "B")),
        False,
    )

    outputs = []
    for thread_count in ("1", "3"):
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = thread_count
        completed = subprocess.run(
            [sys.executable, "-c", _THREAD_PROBE],
            capture_output=True,
            text=True,
            env=env,
            timeout=600,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        outputs.append(json.loads(completed.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1]
    got = torch.tensor(outputs[0]["values"]).reshape(9, 6)
    assert torch.allclose(got, dense_a @ dense_b, atol=1e-3, rtol=1e-3)
