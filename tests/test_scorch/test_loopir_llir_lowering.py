"""LoopIR-to-LLIR lowering: legacy source parity and target boundaries.

The load-bearing gate of the Phase-4 dense slice: for every migrated family
member the LoopIR pipeline's generated C++ must be byte-identical to the
untouched legacy pipeline's C++ for the same normalized CIN program.  These
comparisons run both lowerings in-process without invoking the external C++
compiler, so the whole grid stays fast.

Structural activation assertions cover the target components the lowering
must engage (outer-loop OpenMP policy, hoisted restrict pointers, zero-fill)
even though byte-identical emission already implies them; per the design,
structural activation is never waived.
"""

import copy
from types import MappingProxyType

import pytest
import torch

from scorch.compiler import llir
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
from scorch.compiler.identity import SymbolId
from scorch.compiler.loop_plan import MAX_AFFINE_TILE_WIDTH
from scorch.compiler.llir_pass_manager import LLIRPassManager
from scorch.compiler.loopir.lower_llir import (
    LoopIRTargetError,
    lower_loopir_to_llir,
)
from scorch.compiler.loopir.nodes import ScalarType
from scorch.compiler.loopir.pipeline import (
    compare_generated_sources,
    compile_cin_via_loopir,
)
from scorch.compiler.loopir.verifier import LoopIRVerificationError

from tests.test_scorch.test_loopir_printer import build_matvec
from tests.test_scorch.test_loopir_verifier import (
    build_matmul,
    build_tiled_matvec,
    build_vector_add,
    forge,
)

F32 = torch.float32
F64 = torch.float64


def case_elementwise_add_2d(dtype=F32):
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd", dtype=dtype)
    b = TensorVar("B", fmt="dd", dtype=dtype)
    c = TensorVar("C", fmt="dd", dtype=dtype)
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]))
    return ForAll(i, ForAll(j, assign)), (3, 4), [((3, 4), dtype), ((3, 4), dtype)]


def case_elementwise_mul_2d():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.MUL, a[i, j], b[i, j]))
    return ForAll(i, ForAll(j, assign)), (5, 2), [((5, 2), F32), ((5, 2), F32)]


def case_three_input_fused():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    d = TensorVar("D", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(
        c[i, j],
        CINBinaryOp(
            Operation.MUL,
            CINBinaryOp(Operation.ADD, a[i, j], b[i, j]),
            d[i, j],
        ),
    )
    return ForAll(i, ForAll(j, assign)), (3, 4), [((3, 4), F32)] * 3


def case_vector_add():
    i = IndexVar("i")
    a = TensorVar("a", fmt="d")
    b = TensorVar("b", fmt="d")
    c = TensorVar("c", fmt="d")
    assign = TensorAssign(c[i], CINBinaryOp(Operation.ADD, a[i], b[i]))
    return ForAll(i, assign), (7,), [((7,), F32), ((7,), F32)]


def case_broadcast_row():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("a", fmt="d")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[j])
    return ForAll(i, ForAll(j, assign)), (3, 4), [((4,), F32)]


def case_rank3_elementwise():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a = TensorVar("A", fmt="ddd")
    b = TensorVar("B", fmt="ddd")
    c = TensorVar("C", fmt="ddd")
    assign = TensorAssign(
        c[i, j, k], CINBinaryOp(Operation.ADD, a[i, j, k], b[i, j, k])
    )
    return (
        ForAll(i, ForAll(j, ForAll(k, assign))),
        (2, 3, 4),
        [((2, 3, 4), F32)] * 2,
    )


def case_row_sum():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    y = TensorVar("y", fmt="d")
    assign = TensorAssign(y[i], a[i, j], op=Operation.ADD)
    return ForAll(i, ForAll(j, assign)), (3,), [((3, 4), F32)]


def case_col_sum():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    y = TensorVar("y", fmt="d")
    assign = TensorAssign(y[j], a[i, j], op=Operation.ADD)
    return ForAll(i, ForAll(j, assign)), (4,), [((3, 4), F32)]


def case_matvec():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    x = TensorVar("x", fmt="d")
    y = TensorVar("y", fmt="d")
    assign = TensorAssign(
        y[i], CINBinaryOp(Operation.MUL, a[i, j], x[j]), op=Operation.ADD
    )
    return ForAll(i, ForAll(j, assign)), (3,), [((3, 4), F32), ((4,), F32)]


def case_matmul_ikj(dtype=F32, shapes=((3, 5), (5, 4)), result=(3, 4)):
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    a = TensorVar("A", fmt="dd", dtype=dtype)
    b = TensorVar("B", fmt="dd", dtype=dtype)
    c = TensorVar("C", fmt="dd", dtype=dtype)
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=Operation.ADD
    )
    return (
        ForAll(i, ForAll(k, ForAll(j, assign))),
        result,
        [(shapes[0], dtype), (shapes[1], dtype)],
    )


def case_matmul_f64():
    return case_matmul_ikj(dtype=F64)


def case_matmul_zero_rows():
    return case_matmul_ikj(shapes=((0, 5), (5, 4)), result=(0, 4))


def case_elementwise_zero_extent():
    cin, _, _ = case_elementwise_add_2d()
    return cin, (0, 4), [((0, 4), F32), ((0, 4), F32)]


FAMILY_GRID = [
    ("elementwise_add_2d_f32", case_elementwise_add_2d),
    ("elementwise_add_2d_f64", lambda: case_elementwise_add_2d(F64)),
    ("elementwise_mul_2d", case_elementwise_mul_2d),
    ("three_input_fused", case_three_input_fused),
    ("vector_add", case_vector_add),
    ("broadcast_row", case_broadcast_row),
    ("rank3_elementwise", case_rank3_elementwise),
    ("row_sum", case_row_sum),
    ("col_sum", case_col_sum),
    ("matvec", case_matvec),
    ("matmul_ikj_f32", case_matmul_ikj),
    ("matmul_ikj_f64", case_matmul_f64),
    ("matmul_zero_rows", case_matmul_zero_rows),
    ("elementwise_zero_extent", case_elementwise_zero_extent),
]


@pytest.mark.parametrize(
    "case", [entry[1] for entry in FAMILY_GRID], ids=[e[0] for e in FAMILY_GRID]
)
def test_generated_source_is_byte_identical_to_legacy(case):
    cin, result_shape, bindings = case()
    comparison = compare_generated_sources(cin, result_shape, bindings)
    assert comparison.identical, (
        "LoopIR and legacy generated sources diverged:\n"
        + comparison.loopir_cpp
        + "\n=== legacy ===\n"
        + comparison.legacy_cpp
    )


def test_structural_activation_parallel_and_hoist():
    cin, result_shape, bindings = case_matmul_ikj()
    kernel = compile_cin_via_loopir(cin, result_shape, bindings)
    source = kernel.cpp_source
    assert "#pragma omp parallel for num_threads(scorch_nthreads" in source
    assert "const float* __restrict__ _B_val_ptr" in source
    assert "scorch_zero_dense(C_values, C_capacity);" in source
    assert "C_values[pC1] += A_val[pA1] * _B_val_ptr[j];" in source


def test_reduction_only_outer_loop_is_not_parallel():
    cin, result_shape, bindings = case_col_sum()
    kernel = compile_cin_via_loopir(cin, result_shape, bindings)
    assert "#pragma omp parallel for" not in kernel.cpp_source


def test_loopir_dumps_are_target_neutral():
    cin, result_shape, bindings = case_matmul_ikj()
    kernel = compile_cin_via_loopir(cin, result_shape, bindings)
    for artifact in (kernel.program_text, kernel.program_dump):
        for spelling in ("torch", "omp", "float*", "restrict", "int64_t", "::"):
            assert spelling not in artifact


def test_sub_family_member_lowers_without_legacy_comparand():
    """Dense SUB is outside legacy support (the legacy iteration lattice
    raises NotImplementedError on Operation.SUB), so its coverage is
    LoopIR-only: oracle plus execution tests own its numerics."""

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.SUB, a[i, j], b[i, j]))
    cin = ForAll(i, ForAll(j, assign))
    kernel = compile_cin_via_loopir(cin, (3, 4), [((3, 4), F32), ((3, 4), F32)])
    assert "C_values[pC1] = _A_val_ptr[j] - _B_val_ptr[j];" in kernel.cpp_source


# -- target-boundary fail-closed coverage -----------------------------------


def expect_target_code(code, program, input_shapes, result_shape):
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            program, input_shapes=input_shapes, result_shape=result_shape
        )
    assert error.value.defect.code == code, error.value.defect


def matvec_shapes(program):
    a_symbol, x_symbol = program.inputs
    return {a_symbol: (3, 4), x_symbol: (4,)}


def test_target_lowering_verifies_first():
    fixture = build_vector_add()
    forge(fixture.program.tensors[1], dtype=ScalarType.FLOAT64)
    with pytest.raises(LoopIRVerificationError):
        lower_loopir_to_llir(
            fixture.program,
            input_shapes={fixture.a: (2,), fixture.b: (2,)},
            result_shape=(2,),
        )


def test_target_rejects_tile_widths_that_do_not_fit_constexpr_int():
    fixture = build_tiled_matvec(width=MAX_AFFINE_TILE_WIDTH + 1)
    a_symbol, x_symbol = fixture.program.inputs
    expect_target_code(
        "unsupported_tile_width",
        fixture.program,
        {a_symbol: (3, 4), x_symbol: (4,)},
        (3,),
    )


def test_target_lowering_owns_supplied_context_stage_and_pass_records():
    fixture = build_vector_add()
    options = CompileOptions.from_environment()
    context = CompilationContext(options)
    lower_loopir_to_llir(
        fixture.program,
        input_shapes={fixture.a: (2,), fixture.b: (2,)},
        result_shape=(2,),
        compilation_context=context,
    )
    assert [record.stage_id for record in context.stage_run_records] == [
        CompilerStageId.LOOPIR_TO_LLIR_LOWERING
    ]
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]


def test_invalid_display_name():
    program = build_matvec()
    forge(program.tensors[0], name="A values")
    expect_target_code("invalid_display_name", program, matvec_shapes(program), (3,))


@pytest.mark.parametrize("name", ["for", "λ"])
def test_cpp_unsafe_display_name(name):
    program = build_matvec()
    forge(program.dimensions[0], name=name)
    expect_target_code("invalid_display_name", program, matvec_shapes(program), (3,))


def test_duplicate_display_name():
    program = build_matvec()
    forge(program.tensors[1], name="A")
    expect_target_code("duplicate_display_name", program, matvec_shapes(program), (3,))


def test_dimension_and_tensor_names_share_one_namespace():
    program = build_matvec()
    forge(program.tensors[0], name="i")
    expect_target_code("duplicate_display_name", program, matvec_shapes(program), (3,))


@pytest.mark.parametrize(
    ("mutation", "name"),
    [
        ("dimension", "pA0"),
        ("output", "A_val"),
        ("input", "result"),
    ],
)
def test_generated_cpp_names_cannot_collide(mutation, name):
    program = build_matvec()
    if mutation == "dimension":
        forge(program.dimensions[0], name=name)
    elif mutation == "output":
        forge(program.tensors[-1], name=name)
    else:
        forge(program.tensors[0], name=name)
    expect_target_code(
        "generated_name_collision",
        program,
        matvec_shapes(program),
        (3,),
    )


def test_repeated_operand_loads_fail_at_target_boundary():
    fixture = build_vector_add()
    store = fixture.program.body.statements[0].body.statements[0]
    forge(store.value.rhs, tensor=fixture.a)
    expect_target_code(
        "unsupported_repeated_operand",
        fixture.program,
        {fixture.a: (2,), fixture.b: (2,)},
        (2,),
    )


def test_unsupported_mode_order_at_target_boundary():
    """Compressed structure stays a target mode-order boundary.

    All-dense input and result permutations are level-mapped.  Forging a
    result permutation without rebuilding its loop order therefore reaches
    the physical storage-order diagnostic, while compressed structure stays
    outside the target's admitted layout surface.
    """

    fixture = build_matmul()
    result_decl = fixture.program.tensors[-1]
    assert result_decl.name == "C"
    forge(result_decl.levels[0], mode=1)
    forge(result_decl.levels[1], mode=0)
    expect_target_code(
        "unsupported_loop_order",
        fixture.program,
        {fixture.a: (3, 5), fixture.b: (5, 4)},
        (4, 3),
    )

    # A forged compressed-structure permutation cannot even reach the
    # target: the verifier's dimension-domain rules reject the layout-
    # inconsistent position chain first.  The target's compressed branch
    # stays as defense in depth behind that boundary.
    from scorch.compiler.loopir.lower_cin import lower_normalized_cin_to_loopir

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="ds")
    c = TensorVar("C", fmt="dd")
    cin = ForAll(i, ForAll(j, TensorAssign(c[i, j], a[i, j])))
    lowered = lower_normalized_cin_to_loopir(cin)
    sparse_decl = next(decl for decl in lowered.program.tensors if decl.name == "A")
    forge(sparse_decl.levels[0], mode=1)
    forge(sparse_decl.levels[1], mode=0)
    a_symbol = lowered.program.inputs[0]
    with pytest.raises(LoopIRVerificationError) as error:
        lower_loopir_to_llir(
            lowered.program,
            input_shapes={a_symbol: (3, 4)},
            result_shape=(3, 4),
        )
    assert error.value.defect.code == "domain_mismatch"


def test_permuted_access_metadata_remains_in_logical_index_order():
    """Physical flattening never rewrites stable logical access provenance."""

    from scorch.compiler.loopir.lower_cin import lower_normalized_cin_to_loopir

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    source = TensorVar("A", fmt="ddd", mode_order=[1, 2, 0])
    result = TensorVar("C", fmt="ddd", mode_order=[1, 2, 0])
    source_access = source[i, j, k]
    result_access = result[i, j, k]
    cin = ForAll(
        j,
        ForAll(k, ForAll(i, TensorAssign(result_access, source_access))),
    )
    lowered = lower_normalized_cin_to_loopir(cin)
    function = lower_loopir_to_llir(
        lowered.program,
        input_shapes={lowered.input_symbols[0]: (3, 4, 2)},
        result_shape=(3, 4, 2),
    )

    metadata = {
        node.tensor_access.role: node.tensor_access
        for node in relayout_llir_nodes(function.body)
        if type(node) is llir.ArrayAccess
        and type(node.tensor_access) is llir.TensorAccessMetadata
    }
    assert metadata[llir.TensorAccessRole.INPUT_READ].index_ids == tuple(
        source_access.index_ids
    )
    assert metadata[llir.TensorAccessRole.RESULT_WRITE].index_ids == tuple(
        result_access.index_ids
    )


def test_invalid_shape_binding_coverage_and_rank():
    program = build_matvec()
    a_symbol, x_symbol = program.inputs
    expect_target_code("invalid_shape_binding", program, {a_symbol: (3, 4)}, (3,))
    expect_target_code(
        "invalid_shape_binding",
        program,
        {a_symbol: (3, 4), x_symbol: (4, 1)},
        (3,),
    )
    expect_target_code(
        "invalid_shape_binding",
        program,
        {a_symbol: (3, 4), x_symbol: (4,)},
        (3, 1),
    )
    expect_target_code(
        "invalid_shape_binding",
        program,
        {a_symbol: (3, -4), x_symbol: (4,)},
        (3,),
    )


def test_dimension_extent_mismatch():
    program = build_matvec()
    a_symbol, x_symbol = program.inputs
    expect_target_code(
        "dimension_extent_mismatch",
        program,
        {a_symbol: (3, 4), x_symbol: (5,)},
        (3,),
    )
    expect_target_code(
        "dimension_extent_mismatch",
        program,
        {a_symbol: (3, 4), x_symbol: (4,)},
        (9,),
    )


def test_unsupported_program_shape_multi_statement_block():
    fixture = build_vector_add()
    store = fixture.program.body.statements[0].body.statements[0]
    loop = fixture.program.body.statements[0]
    from scorch.compiler.loopir.nodes import Block, LoopIRNodeId

    duplicate = Block(LoopIRNodeId(12_000), (store,))
    del duplicate
    forge(
        loop,
        body=Block(
            LoopIRNodeId(12_001),
            (
                store,
                Block(LoopIRNodeId(12_002), ()),
            ),
        ),
    )
    expect_target_code(
        "unsupported_program_shape",
        fixture.program,
        {fixture.a: (2,), fixture.b: (2,)},
        (2,),
    )


# -- Phase-5 sparse families: byte parity and target boundaries ---------------

from tests.test_scorch.test_loopir_verifier import (  # noqa: E402
    build_csr_spmv,
    build_union_add,
    forge as _forge,
)


def case_spmv(dtype=F32):
    i, j = IndexVar("i"), IndexVar("j")
    y = TensorVar("y", fmt="d", dtype=dtype)
    a = TensorVar("A", fmt="ds", dtype=dtype)
    x = TensorVar("x", fmt="d", dtype=dtype)
    assign = TensorAssign(
        y[i], CINBinaryOp(Operation.MUL, a[i, j], x[j]), op=Operation.ADD
    )
    cin = ForAll(i, ForAll(j, assign))
    return cin, (5,), [((5, 7), dtype), ((7,), dtype)]


def case_spmv_f64():
    return case_spmv(torch.float64)


def case_spmm_csr_dense():
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=Operation.ADD
    )
    cin = ForAll(i, ForAll(k, ForAll(j, assign)))
    return cin, (5, 4), [((5, 7), F32), ((7, 4), F32)]


def case_sparse_elementwise(op, fmt_out):
    def build():
        i, j = IndexVar("i"), IndexVar("j")
        c = TensorVar("C", fmt=fmt_out)
        a = TensorVar("A", fmt="ds")
        b = TensorVar("B", fmt="ds")
        assign = TensorAssign(c[i, j], CINBinaryOp(op, a[i, j], b[i, j]))
        cin = ForAll(i, ForAll(j, assign))
        return cin, (5, 7), [((5, 7), F32), ((5, 7), F32)]

    return build


def case_csr_row_sum():
    i, j = IndexVar("i"), IndexVar("j")
    y = TensorVar("y", fmt="d")
    a = TensorVar("A", fmt="ds")
    assign = TensorAssign(y[i], a[i, j], op=Operation.ADD)
    cin = ForAll(i, ForAll(j, assign))
    return cin, (5,), [((5, 7), F32)]


def case_sampled_elementwise_dd():
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.MUL, a[i, j], b[i, j]))
    cin = ForAll(i, ForAll(j, assign))
    return cin, (5, 7), [((5, 7), F32), ((5, 7), F32)]


def case_spgemm_to_dense():
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=Operation.ADD
    )
    cin = ForAll(i, ForAll(k, ForAll(j, assign)))
    return cin, (5, 4), [((5, 7), F32), ((7, 4), F32)]


def case_union_times_dense():
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    d = TensorVar("D", fmt="dd")
    assign = TensorAssign(
        c[i, j],
        CINBinaryOp(
            Operation.MUL, CINBinaryOp(Operation.ADD, a[i, j], b[i, j]), d[i, j]
        ),
    )
    cin = ForAll(i, ForAll(j, assign))
    return cin, (5, 7), [((5, 7), F32)] * 3


SPARSE_FAMILY_GRID = [
    ("spmv_csr_f32", case_spmv),
    ("spmv_csr_f64", case_spmv_f64),
    ("spmm_csr_dense", case_spmm_csr_dense),
    ("union_add_to_csr", case_sparse_elementwise(Operation.ADD, "ds")),
    ("union_add_to_dense", case_sparse_elementwise(Operation.ADD, "dd")),
    ("intersection_mul_to_csr", case_sparse_elementwise(Operation.MUL, "ds")),
    ("intersection_mul_to_dense", case_sparse_elementwise(Operation.MUL, "dd")),
    ("csr_row_sum", case_csr_row_sum),
    ("sampled_elementwise_dd", case_sampled_elementwise_dd),
    ("spgemm_to_dense", case_spgemm_to_dense),
    ("union_times_dense", case_union_times_dense),
]


@pytest.mark.parametrize(
    "case",
    [entry[1] for entry in SPARSE_FAMILY_GRID],
    ids=[entry[0] for entry in SPARSE_FAMILY_GRID],
)
def test_sparse_generated_source_is_byte_identical_to_legacy(case):
    cin, result_shape, bindings = case()
    comparison = compare_generated_sources(cin, result_shape, bindings)
    assert comparison.identical, (
        "LoopIR and legacy generated sources diverged:\n"
        + comparison.loopir_cpp
        + "\n=== legacy ===\n"
        + comparison.legacy_cpp
    )


def test_sparse_structural_activation_spmm():
    """Prefetch, pointer hoist, position loop, and the sparse-aware parallel
    policy must all fire on the CSR-by-dense SpMM kernel."""

    cin, result_shape, bindings = case_spmm_csr_dense()
    kernel = compile_cin_via_loopir(cin, result_shape, bindings)
    source = kernel.cpp_source
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(A1_pos[A0_size], "
        "A0_size)) schedule(dynamic, scorch_chunk(A0_size, A1_pos[A0_size]))"
    ) in source
    assert "__builtin_prefetch(&B_val[A1_crd[pA1 + 1] * B1_size], 0, 1);" in source
    assert "const float* __restrict__ _B_val_ptr = &B_val[pB0 * B1_size];" in source
    assert "for (int pA1 = A1_pos[pA0]; pA1 < pA1_end; pA1++)" in source


def test_sparse_structural_activation_union_assembly():
    """The CSR union kernel owns ordered assembly and stays serial."""

    build = case_sparse_elementwise(Operation.ADD, "ds")
    cin, result_shape, bindings = build()
    kernel = compile_cin_via_loopir(cin, result_shape, bindings)
    source = kernel.cpp_source
    assert "#pragma omp parallel for" not in source
    assert "scorch_vector_set(C1_pos, 0, 0);" in source
    assert "while (pA1 < pA1_end && pB1 < pB1_end)" in source
    assert "int j = std::min({j_A, j_B});" in source
    assert "} else if (j_A == j) {" in source
    assert "pA1 += (int)(j_A == j);" in source
    assert "while (pA1 < pA1_end) {" in source
    assert "while (pB1 < pB1_end) {" in source
    assert "scorch_tensor_from_vector(std::move(C1_pos), torch::kInt);" in source


def test_sparse_structural_activation_merged_dense_policy():
    """Merged-into-dense kernels keep the legacy row-count-only thread cap."""

    build = case_sparse_elementwise(Operation.ADD, "dd")
    cin, result_shape, bindings = build()
    kernel = compile_cin_via_loopir(cin, result_shape, bindings)
    source = kernel.cpp_source
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(-1, A0_size)) "
        "schedule(dynamic, scorch_chunk(A0_size, -1))"
    ) in source
    assert "if (j_A == j && j_B == j) {" in source


def test_sparse_intersection_emits_single_alignment_case():
    build = case_sparse_elementwise(Operation.MUL, "ds")
    cin, result_shape, bindings = build()
    kernel = compile_cin_via_loopir(cin, result_shape, bindings)
    source = kernel.cpp_source
    assert "if (j_A == j && j_B == j) {" in source
    assert "else if" not in source
    assert "while (pA1 < pA1_end) {" not in source.split("&&")[-1]


def test_sparse_loopir_dumps_are_target_neutral():
    build = case_sparse_elementwise(Operation.ADD, "ds")
    cin, result_shape, bindings = build()
    kernel = compile_cin_via_loopir(cin, result_shape, bindings)
    for artifact in (kernel.program_text, kernel.program_dump):
        for spelling in (
            "torch",
            "omp_",
            "#pragma",
            "float*",
            "restrict",
            "int64_t",
            "::",
        ):
            assert spelling not in artifact


def union_shapes(program):
    a_symbol, b_symbol = program.inputs
    return {a_symbol: (3, 4), b_symbol: (3, 4)}


def test_level_zero_cursors_fail_at_target_boundary():
    """DCSR descent verifies and lowers, but the target fails closed."""

    i, j = IndexVar("i"), IndexVar("j")
    y = TensorVar("y", fmt="d")
    a = TensorVar("A", fmt="ss")
    x = TensorVar("x", fmt="d")
    assign = TensorAssign(
        y[i], CINBinaryOp(Operation.MUL, a[i, j], x[j]), op=Operation.ADD
    )
    cin = ForAll(i, ForAll(j, assign))
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(cin, (5,), [((5, 7), F32), ((7,), F32)])
    assert error.value.defect.code == "unsupported_program_shape"


def test_three_cursor_merge_fails_at_target_boundary():
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="ds")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    d = TensorVar("D", fmt="ds")
    assign = TensorAssign(
        c[i, j],
        CINBinaryOp(
            Operation.ADD, CINBinaryOp(Operation.ADD, a[i, j], b[i, j]), d[i, j]
        ),
    )
    cin = ForAll(i, ForAll(j, assign))
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(cin, (5, 7), [((5, 7), F32)] * 3)
    assert error.value.defect.code == "unsupported_program_shape"


def test_append_without_merge_fails_at_target_boundary():
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="ds")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.MUL, a[i, j], b[i, j]))
    cin = ForAll(i, ForAll(j, assign))
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(cin, (5, 7), [((5, 7), F32), ((5, 7), F32)])
    assert error.value.defect.code == "unsupported_program_shape"


def test_nonzero_union_default_fails_at_target_boundary():
    fixture = build_union_add()
    merged = fixture.program.body.statements[0].body.statements[0]
    leaf = merged.body.statements[0]
    _forge(leaf.value.lhs, default=fixture.builder.float_const(1.5))
    expect_target_code(
        "unsupported_union_default",
        fixture.program,
        union_shapes(fixture.program),
        (3, 4),
    )


def test_unread_input_fails_at_target_boundary():
    fixture = build_csr_spmv()
    leaf = fixture.program.body.statements[0].body.statements[0].body.statements[0]
    _forge(leaf, value=leaf.value.lhs)
    expect_target_code(
        "unsupported_program_shape",
        fixture.program,
        {fixture.a: (3, 4), fixture.x: (4,)},
        (3,),
    )


def test_sparse_generated_names_are_reserved():
    # The merged coordinate temporary for input tensor 'A' spells 'j_A';
    # a tensor claiming that display name must be rejected.
    fixture = build_union_add()
    decl_b = fixture.program.tensors[1]
    _forge(decl_b, name="j_A")
    expect_target_code(
        "generated_name_collision",
        fixture.program,
        union_shapes(fixture.program),
        (3, 4),
    )


def test_sparse_pos_names_are_reserved():
    fixture = build_csr_spmv()
    decl_x = fixture.program.tensors[1]
    _forge(decl_x, name="A1_pos")
    expect_target_code(
        "generated_name_collision",
        fixture.program,
        {fixture.a: (3, 4), fixture.x: (4,)},
        (3,),
    )


def test_sparse_target_owns_context_stage_and_pass_records():
    """The sparse path records the same owned stage and managed-pass runs."""

    cin, result_shape, bindings = case_spmm_csr_dense()
    options = CompileOptions.from_environment(environ={})
    context = CompilationContext(options)
    compile_cin_via_loopir(
        cin,
        result_shape,
        bindings,
        compile_options=options,
        compilation_context=context,
    )
    stage_ids = [record.stage_id for record in context.stage_run_records]
    assert CompilerStageId.CIN_TO_LOOPIR_LOWERING in stage_ids
    assert CompilerStageId.LOOPIR_TO_LLIR_LOWERING in stage_ids
    pass_names = [record.pass_name for record in context.llir_pass_run_records]
    assert "insert_sparse_prefetch" in pass_names
    assert "hoist_dense_pointers" in pass_names


# -- Phase-6 workspace regions (target boundary) ------------------------------

from tests.test_scorch.test_loopir_verifier import (  # noqa: E402
    build_stack_matmul,
)


def stack_matmul_shapes(fixture, rows=3, inner=4, cols=5):
    return (
        {fixture.a: (rows, inner), fixture.b: (inner, cols)},
        (rows, cols),
    )


def _expect_post_construction_graph_rejection(
    program,
    input_shapes,
    result_shape,
    mutate,
    *,
    reverify=True,
    message="program graph",
):
    """Exercise the retained target directly across its mutation window."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module
    from scorch.compiler.loopir.verifier import verify_program

    target = lower_llir_module._TargetLowering(
        program,
        input_shapes,
        result_shape,
    )
    mutate(target)
    if reverify:
        verify_program(program)
    with pytest.raises(LoopIRTargetError) as error:
        target.raw_loop_statements()
    assert error.value.defect.code == "unsupported_program_shape"
    assert message in error.value.defect.message


@pytest.mark.parametrize(
    "mutation",
    ["root", "ancestor", "leaf", "input_decl", "result_decl"],
    ids=str,
)
def test_target_binds_the_complete_dense_program_graph(mutation):
    """Fresh verifier-valid graph twins cannot leave retained owners stale."""

    from scorch.compiler.loopir.build import LoopIRBuilder

    fixture = build_matmul()
    shapes = {fixture.a: (2, 3), fixture.b: (3, 4)}

    def mutate(target):
        if mutation == "root":
            object.__setattr__(
                target.program, "body", copy.deepcopy(target.program.body)
            )
            return
        if mutation == "ancestor":
            outer = target.loops[0].node
            object.__setattr__(outer, "body", copy.deepcopy(outer.body))
            return
        if mutation == "leaf":
            body = target.loops[-1].node.body
            object.__setattr__(body, "statements", (copy.deepcopy(target.leaf),))
            return
        tensor_index = 0 if mutation == "input_decl" else 2
        old = target.program.tensors[tensor_index]
        builder = LoopIRBuilder.resuming(target.program)
        fresh = builder.tensor(
            old.symbol,
            old.name,
            old.dtype,
            old.dimensions,
            old.levels,
        )
        tensors = list(target.program.tensors)
        tensors[tensor_index] = fresh
        object.__setattr__(target.program, "tensors", tuple(tensors))

    _expect_post_construction_graph_rejection(
        fixture.program,
        shapes,
        (2, 4),
        mutate,
    )


def test_target_graph_snapshot_strongly_owns_replaced_root():
    """A freed graph object cannot recycle its address into the signature."""

    import gc
    import weakref

    from scorch.compiler.loopir import lower_llir as lower_llir_module
    from scorch.compiler.loopir.verifier import verify_program

    fixture = build_matmul()
    target = lower_llir_module._TargetLowering(
        fixture.program,
        {fixture.a: (2, 3), fixture.b: (3, 4)},
        (2, 4),
    )
    original = fixture.program.body
    original_ref = weakref.ref(original)
    fresh = copy.deepcopy(original)
    object.__setattr__(fixture.program, "body", fresh)
    del original
    gc.collect()
    assert original_ref() is not None
    assert any(owner is original_ref() for owner in target._program_graph_owners)
    verify_program(fixture.program)
    with pytest.raises(LoopIRTargetError) as error:
        target.raw_loop_statements()
    assert error.value.defect.code == "unsupported_program_shape"
    assert "program graph" in error.value.defect.message


def test_target_graph_rejects_hostile_class_descriptor_without_invoking_it():
    """Foreign stored values cannot execute class/metaclass hooks."""

    class ClassBombMeta(type):
        def __eq__(cls, other):
            raise RuntimeError("hostile metaclass equality executed")

        def __hash__(cls):
            raise RuntimeError("hostile metaclass hash executed")

    class ClassBomb(metaclass=ClassBombMeta):
        @property
        def __class__(self):
            raise RuntimeError("hostile __class__ descriptor executed")

    fixture = build_matmul()

    def mutate(target):
        object.__setattr__(target.leaf, "value", ClassBomb())

    _expect_post_construction_graph_rejection(
        fixture.program,
        {fixture.a: (2, 3), fixture.b: (3, 4)},
        (2, 4),
        mutate,
        reverify=False,
        message="unsupported target value",
    )


def test_target_graph_rejects_hostile_identity_key_without_equality():
    """Malformed identity dictionaries cannot execute key comparison hooks."""

    class KeyBomb:
        def __hash__(self):
            return hash("value")

        def __eq__(self, other):
            raise RuntimeError("hostile identity-key equality executed")

    fixture = build_matmul()

    def mutate(target):
        state = object.__getattribute__(target.leaf.node_id, "__dict__")
        state.clear()
        state[KeyBomb()] = 1

    _expect_post_construction_graph_rejection(
        fixture.program,
        {fixture.a: (2, 3), fixture.b: (3, 4)},
        (2, 4),
        mutate,
        reverify=False,
        message="malformed node identity",
    )


def test_target_graph_registry_is_exhaustive_for_the_loopir_schema():
    """Every new schema node or enum must explicitly enter the graph guard."""

    from enum import EnumMeta

    from scorch.compiler.loopir import lower_llir as lower_llir_module
    from scorch.compiler.loopir import nodes as loopir_nodes

    node_types = {
        candidate
        for candidate in vars(loopir_nodes).values()
        if type(candidate) is type
        and candidate.__module__ == loopir_nodes.__name__
        and issubclass(candidate, loopir_nodes.LoopIRNode)
        and candidate
        not in {loopir_nodes.LoopIRNode, loopir_nodes.Expr, loopir_nodes.Stmt}
    }
    enum_types = {
        candidate
        for candidate in vars(loopir_nodes).values()
        if isinstance(candidate, EnumMeta)
        and candidate.__module__ == loopir_nodes.__name__
    }
    assert set(lower_llir_module._LOOPIR_GRAPH_NODE_TYPES) == node_types
    assert set(lower_llir_module._LOOPIR_GRAPH_ENUM_TYPES) == enum_types
    assert set(lower_llir_module._LOOPIR_NODE_TYPE_BY_ID.values()) == node_types
    assert set(lower_llir_module._LOOPIR_ENUM_TYPE_BY_ID.values()) == enum_types


def test_unchanged_target_skips_redundant_narrow_integrity_scans(monkeypatch):
    """The complete graph guard is the successful-path ownership proof."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = build_matmul()
    target = lower_llir_module._TargetLowering(
        fixture.program,
        {fixture.a: (2, 3), fixture.b: (3, 4)},
        (2, 4),
    )

    def redundant_scan(*args, **kwargs):
        raise AssertionError("an unchanged graph must not repeat narrow scans")

    for name in (
        "_validated_bound_position_bindings",
        "_validated_value_expression_signature",
        "_validated_target_owner_signature",
    ):
        monkeypatch.setattr(
            lower_llir_module._TargetLowering,
            name,
            redundant_scan,
        )

    assert target.raw_loop_statements()


def test_diagnostic_witness_replacement_cannot_mask_a_graph_change():
    """Frozen diagnostic caches cannot become an integrity authority."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = build_matmul()
    target = lower_llir_module._TargetLowering(
        fixture.program,
        {fixture.a: (2, 3), fixture.b: (3, 4)},
        (2, 4),
    )
    object.__setattr__(target.leaf, "value", copy.deepcopy(target.leaf.value))

    # Forge every narrow witness with an exact immutable replacement.  The
    # witnesses retain specific diagnostics after the complete graph guard
    # detects a change; none may authorize emission or replace its sealed
    # constructor-owned container.
    object.__setattr__(
        target,
        "_bound_position_snapshot",
        MappingProxyType(target._validated_bound_position_bindings()),
    )
    object.__setattr__(
        target,
        "_position_load_signatures",
        MappingProxyType(dict(target._position_load_signatures)),
    )
    object.__setattr__(
        target,
        "_value_expression_snapshot",
        tuple(
            list(
                target._validated_value_expression_signature(
                    target._access_value_expression()
                )
            )
        ),
    )
    object.__setattr__(
        target,
        "_target_owner_snapshot",
        tuple(list(target._validated_target_owner_signature())),
    )

    with pytest.raises(LoopIRTargetError) as error:
        target.raw_loop_statements()
    assert error.value.defect.code == "unsupported_program_shape"
    assert "retained program caches" in error.value.defect.message


@pytest.mark.parametrize(
    "field",
    [
        "_bound_position_snapshot",
        "_position_load_signatures",
        "_value_expression_snapshot",
        "_target_owner_snapshot",
    ],
)
def test_target_rejects_hostile_diagnostic_witness_before_using_it(field):
    """A replaced witness cannot run callbacks at a narrow integrity check."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    class CallbackBomb:
        def __eq__(self, other):
            raise RuntimeError("hostile witness comparison executed")

    class CallbackBombDict(dict):
        def __len__(self):
            raise RuntimeError("hostile witness length executed")

        def get(self, *args, **kwargs):
            raise RuntimeError("hostile witness lookup executed")

    fixture = build_matmul()
    target = lower_llir_module._TargetLowering(
        fixture.program,
        {fixture.a: (2, 3), fixture.b: (3, 4)},
        (2, 4),
    )
    replacement = (
        MappingProxyType(CallbackBombDict())
        if field in {"_bound_position_snapshot", "_position_load_signatures"}
        else (CallbackBomb(),)
    )
    object.__setattr__(target, field, replacement)

    with pytest.raises(LoopIRTargetError) as error:
        target.raw_loop_statements()
    assert error.value.defect.code == "unsupported_program_shape"
    assert "retained program caches" in error.value.defect.message


@pytest.mark.parametrize("mutation", ["replace", "delete"], ids=str)
def test_target_fails_closed_when_its_program_reference_changes(mutation):
    fixture = build_matmul()
    from scorch.compiler.loopir import lower_llir as lower_llir_module

    target = lower_llir_module._TargetLowering(
        fixture.program,
        {fixture.a: (2, 3), fixture.b: (3, 4)},
        (2, 4),
    )
    if mutation == "replace":
        object.__setattr__(target, "program", object())
    else:
        object.__delattr__(target, "program")
    with pytest.raises(LoopIRTargetError) as error:
        target.raw_loop_statements()
    assert error.value.defect.code == "unsupported_program_shape"
    assert "program reference" in error.value.defect.message


@pytest.mark.parametrize("mutation", ["loop", "result_decl"], ids=str)
def test_target_binds_program_derived_caches_before_emission(mutation):
    """A pristine program cannot be interpreted through rewritten caches."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = build_matmul()
    target = lower_llir_module._TargetLowering(
        fixture.program,
        {fixture.a: (2, 3), fixture.b: (3, 4)},
        (2, 4),
    )
    if mutation == "loop":
        target.loops[0] = target.loops[-1]
    else:
        target.result_decl = target.decls[fixture.a]
    with pytest.raises(LoopIRTargetError) as error:
        target.raw_loop_statements()
    assert error.value.defect.code == "unsupported_program_shape"
    assert "program caches" in error.value.defect.message


@pytest.mark.parametrize("mutation", ["delete_region", "malformed_loop"], ids=str)
def test_target_validates_caches_before_interpreting_them(mutation):
    """Malformed retained caches fail closed without leaking attribute errors."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = build_matmul()
    target = lower_llir_module._TargetLowering(
        fixture.program,
        {fixture.a: (2, 3), fixture.b: (3, 4)},
        (2, 4),
    )
    if mutation == "delete_region":
        object.__delattr__(target, "region")
    else:
        target.loops = [object()]
    with pytest.raises(LoopIRTargetError) as error:
        target.raw_loop_statements()
    assert error.value.defect.code == "unsupported_program_shape"
    assert "program caches" in error.value.defect.message


def test_target_class_replacement_cannot_override_the_integrity_guard():
    """A swapped subclass cannot dynamically replace a base boundary check."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    hostile_callbacks = []

    class HostileMeta(type):
        def __instancecheck__(cls, instance):
            hostile_callbacks.append("instancecheck")
            raise RuntimeError("hostile metaclass callback executed")

        def __subclasscheck__(cls, subclass):
            hostile_callbacks.append("subclasscheck")
            raise RuntimeError("hostile metaclass callback executed")

    class HostileTarget(lower_llir_module._TargetLowering, metaclass=HostileMeta):
        def _require_program_graph_unchanged(self):
            hostile_callbacks.append("graph guard")

        def _require_value_expression_unchanged(self):
            hostile_callbacks.append("value guard")

        def _lower_dense(self, position):
            hostile_callbacks.append("lower dense")
            raise RuntimeError("hostile lowering callback executed")

    fixture = build_matmul()
    target = lower_llir_module._TargetLowering(
        fixture.program,
        {fixture.a: (2, 3), fixture.b: (3, 4)},
        (2, 4),
    )
    object.__setattr__(target, "__class__", HostileTarget)

    with pytest.raises(LoopIRTargetError) as error:
        lower_llir_module._TargetLowering.raw_loop_statements(target)
    assert error.value.defect.code == "unsupported_program_shape"
    assert "program caches" in error.value.defect.message
    assert hostile_callbacks == []


def test_target_binds_synthetic_cached_loopir_nodes_by_state():
    """Synthetic relayout views are not covered by the program-graph walk."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = relayout_program()
    target = lower_llir_module._TargetLowering(
        fixture.program,
        relayout_shapes(fixture),
        (4, 6),
    )
    view = next(iter(target._staged_views.values()))
    object.__setattr__(view, "tensor", fixture.a)

    with pytest.raises(LoopIRTargetError) as error:
        target.raw_loop_statements()
    assert error.value.defect.code == "unsupported_program_shape"
    assert "program caches" in error.value.defect.message


@pytest.mark.parametrize("field", ["name", "dtype"], ids=str)
def test_target_fails_closed_on_malformed_retained_tensor_declaration(field):
    """Malformed post-construction declarations cannot leak target errors."""

    fixture = build_matmul()

    def mutate(target):
        declaration = target.program.tensors[-1]
        if field == "name":
            object.__delattr__(declaration, "name")
        else:
            object.__setattr__(declaration, "dtype", "float")

    _expect_post_construction_graph_rejection(
        fixture.program,
        {fixture.a: (2, 3), fixture.b: (3, 4)},
        (2, 4),
        mutate,
        reverify=False,
    )


def test_stack_region_lowers_through_the_target():
    fixture = build_stack_matmul(width=4)
    shapes, result_shape = stack_matmul_shapes(fixture)
    function = lower_loopir_to_llir(
        fixture.program, input_shapes=shapes, result_shape=result_shape
    )
    assert function is not None


def test_workspace_target_rejects_post_construction_consumer_replacement(monkeypatch):
    """The actual consumer owner cannot diverge from its retained target view."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module
    from scorch.compiler.loopir.build import LoopIRBuilder
    from scorch.compiler.loopir.nodes import BinaryOp
    from scorch.compiler.loopir.verifier import verify_program

    fixture = build_stack_matmul(width=4)
    shapes, result_shape = stack_matmul_shapes(fixture)
    original = lower_llir_module._TargetLowering.raw_loop_statements

    def replacing(self):
        assert self._region_leaf is not None
        builder = LoopIRBuilder.resuming(self.program)
        object.__setattr__(
            self._region_leaf,
            "value",
            builder.binary(
                BinaryOp.ADD,
                self._region_leaf.value,
                builder.float_const(1.0),
            ),
        )
        verify_program(self.program)
        return original(self)

    monkeypatch.setattr(
        lower_llir_module._TargetLowering,
        "raw_loop_statements",
        replacing,
    )
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            fixture.program,
            input_shapes=shapes,
            result_shape=result_shape,
        )
    assert error.value.defect.code == "unsupported_program_shape"
    assert "owning statement" in error.value.defect.message


@pytest.mark.parametrize(
    "mutation",
    ["producer", "consumer", "workspace_decl"],
    ids=str,
)
def test_workspace_target_binds_both_region_branches_and_declaration(mutation):
    """The current graph must still reach the exact retained workspace state."""

    from scorch.compiler.loopir.build import LoopIRBuilder

    fixture = build_stack_matmul(width=4)
    shapes, result_shape = stack_matmul_shapes(fixture)

    def mutate(target):
        assert target.region is fixture.region
        if mutation == "producer":
            object.__setattr__(
                fixture.region,
                "producer",
                copy.deepcopy(fixture.region.producer),
            )
            return
        if mutation == "consumer":
            object.__setattr__(
                fixture.region,
                "consumer",
                copy.deepcopy(fixture.region.consumer),
            )
            return
        old = fixture.region.workspace
        builder = LoopIRBuilder.resuming(fixture.program)
        fresh = builder.workspace_decl(
            old.workspace,
            old.name,
            old.dtype,
            old.tile,
        )
        object.__setattr__(fixture.region, "workspace", fresh)

    _expect_post_construction_graph_rejection(
        fixture.program,
        shapes,
        result_shape,
        mutate,
    )


def test_workspace_producer_needs_a_reduction_loop():
    """A bare point-loop producer is verifier-legal but outside the family."""

    from scorch.compiler.loopir.build import LoopIRBuilder
    from scorch.compiler.loopir.nodes import ReduceOp, ScalarType
    from scorch.compiler.loopir.verifier import verify_program

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_k = builder.dimension("k")
    x, c = builder.new_symbol_id(), builder.new_symbol_id()
    decl_x = builder.tensor(
        x, "x", ScalarType.FLOAT32, (dim_k.dimension,), builder.dense_levels(1)
    )
    decl_c = builder.tensor(
        c,
        "C",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_k.dimension),
        builder.dense_levels(2),
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    tile = builder.new_tile_id()
    workspace = builder.new_workspace_id()
    decl_w = builder.workspace_decl(workspace, "wksp", ScalarType.FLOAT32, tile)
    producer_point = builder.tile_inner_for(
        tile,
        col,
        dim_k.dimension,
        4,
        False,
        builder.block(
            (
                builder.workspace_reduce(
                    workspace,
                    builder.index_value(col),
                    ReduceOp.ADD,
                    builder.load(x, (builder.index_value(col),)),
                ),
            )
        ),
    )
    consumer_point = builder.tile_inner_for(
        tile,
        col,
        dim_k.dimension,
        4,
        False,
        builder.block(
            (
                builder.store_reduce(
                    c,
                    (builder.index_value(row), builder.index_value(col)),
                    ReduceOp.ADD,
                    builder.workspace_read(workspace, builder.index_value(col)),
                ),
            )
        ),
    )
    region = builder.workspace_region(
        decl_w, builder.block((producer_point,)), builder.block((consumer_point,))
    )
    row_loop = builder.dense_for(row, dim_i.dimension, builder.block((region,)))
    outer = builder.tile_outer_for(
        tile, col, dim_k.dimension, 4, builder.block((row_loop,))
    )
    program = builder.program(
        (dim_i, dim_k), (decl_x, decl_c), (x,), (c,), builder.block((outer,))
    )
    verify_program(program)
    expect_target_code("unsupported_program_shape", program, {x: (5,)}, (3, 5))


def test_workspace_consumer_must_be_the_exact_copy_out_form():
    from scorch.compiler.loopir.build import LoopIRBuilder
    from scorch.compiler.loopir.nodes import BinaryOp, ReduceOp
    from scorch.compiler.loopir.verifier import verify_program

    fixture = build_stack_matmul(width=4)
    builder = LoopIRBuilder.resuming(fixture.program)
    scaled = builder.store_reduce(
        fixture.copy_out.tensor,
        fixture.copy_out.indices,
        ReduceOp.ADD,
        builder.binary(
            BinaryOp.ADD,
            fixture.copy_out.value,
            builder.float_const(0.0),
        ),
    )
    forge(fixture.consumer_inner, body=builder.block((scaled,)))
    verify_program(fixture.program)
    shapes, result_shape = stack_matmul_shapes(fixture)
    expect_target_code(
        "unsupported_program_shape", fixture.program, shapes, result_shape
    )


def test_workspace_target_rejects_distinct_indices_sharing_a_dimension():
    """Dimension identity is not a safe C++ loop-variable identity.

    The semantic IR may iterate one coordinate domain with two independent
    binders.  This target names loop variables from the dimension, so it must
    reject that shape instead of shadowing the reduction coordinate with the
    tiled point coordinate.
    """

    from scorch.compiler.loopir.build import LoopIRBuilder
    from scorch.compiler.loopir.nodes import ReduceOp
    from scorch.compiler.loopir.verifier import verify_program

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_k = builder.dimension("k")
    x, c = builder.new_symbol_id(), builder.new_symbol_id()
    decl_x = builder.tensor(
        x, "X", ScalarType.FLOAT32, (dim_k.dimension,), builder.dense_levels(1)
    )
    decl_c = builder.tensor(
        c,
        "C",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_k.dimension),
        builder.dense_levels(2),
    )
    row = builder.new_index_id()
    reduction = builder.new_index_id()
    point = builder.new_index_id()
    tile = builder.new_tile_id()
    workspace = builder.new_workspace_id()
    workspace_decl = builder.workspace_decl(workspace, "wksp", ScalarType.FLOAT32, tile)
    producer_point = builder.tile_inner_for(
        tile,
        point,
        dim_k.dimension,
        2,
        False,
        builder.block(
            (
                builder.workspace_reduce(
                    workspace,
                    builder.index_value(point),
                    ReduceOp.ADD,
                    builder.load(x, (builder.index_value(reduction),)),
                ),
            )
        ),
    )
    producer = builder.dense_for(
        reduction,
        dim_k.dimension,
        builder.block((producer_point,)),
    )
    copy_out = builder.store_reduce(
        c,
        (builder.index_value(row), builder.index_value(point)),
        ReduceOp.ADD,
        builder.workspace_read(workspace, builder.index_value(point)),
    )
    consumer_point = builder.tile_inner_for(
        tile,
        point,
        dim_k.dimension,
        2,
        False,
        builder.block((copy_out,)),
    )
    region = builder.workspace_region(
        workspace_decl,
        builder.block((producer,)),
        builder.block((consumer_point,)),
    )
    row_loop = builder.dense_for(row, dim_i.dimension, builder.block((region,)))
    outer = builder.tile_outer_for(
        tile,
        point,
        dim_k.dimension,
        2,
        builder.block((row_loop,)),
    )
    program = builder.program(
        (dim_i, dim_k),
        (decl_x, decl_c),
        (x,),
        (c,),
        builder.block((outer,)),
    )
    verify_program(program)
    expect_target_code(
        "generated_name_collision",
        program,
        {x: (3,)},
        (1, 3),
    )


def test_workspace_name_participates_in_generated_name_collisions():
    fixture = build_stack_matmul(width=4)
    forge(fixture.region.workspace, name="B")
    shapes, result_shape = stack_matmul_shapes(fixture)
    expect_target_code(
        "generated_name_collision", fixture.program, shapes, result_shape
    )
    fixture = build_stack_matmul(width=4)
    forge(fixture.region.workspace, name="for")
    shapes, result_shape = stack_matmul_shapes(fixture)
    expect_target_code("invalid_display_name", fixture.program, shapes, result_shape)


# -- Phase-6 sparse panel windows ---------------------------------------------


def panel_spmm_shapes(fixture):
    return {fixture.a: (4, 5), fixture.b: (5, 6)}, (4, 6)


def scheduled_panel_cpp(width=3, dtype=F32):
    from scorch.compiler.scheduler import Schedule, TileSpec

    ivs = {name: IndexVar(name) for name in ("i", "j", "k")}
    c = TensorVar("C", fmt="dd", dtype=dtype)
    a = TensorVar("A", fmt="ds", dtype=dtype)
    b = TensorVar("B", fmt="dd", dtype=dtype)
    stmt = TensorAssign(
        c[ivs["i"], ivs["k"]],
        CINBinaryOp(Operation.MUL, a[ivs["i"], ivs["j"]], b[ivs["j"], ivs["k"]]),
        op=Operation.ADD,
    )
    for name in reversed(("i", "j", "k")):
        stmt = ForAll(ivs[name], stmt)
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(TileSpec("j", width, kind="panel", accum="direct"),),
        tag="panel-activation",
        parallel_loop="i",
    )
    options = CompileOptions.from_environment(requested_schedule=schedule)
    bindings = (((4, 5), dtype), ((5, 6), dtype))
    return compile_cin_via_loopir(
        stmt, (4, 6), bindings, compile_options=options
    ).cpp_source


def test_panel_structural_activation_is_never_waived():
    """The emitted panel kernel must engage every legacy panel component:
    the top-of-function width constant, the serial origin loop over the
    declared dense bound, the clamped window end, both lower_bound-derived
    position bounds, the windowed loop start, the nnz-aware parallel row
    policy, prefetch survival inside the window, and the hoisted operand
    pointer."""

    cpp = scheduled_panel_cpp(width=3)
    body_start = cpp.index("{")
    prologue = cpp[body_start : cpp.index("scorch_native::validate_jit_result_shape")]
    assert "// Initialize j panel tile size" in prologue
    assert "constexpr int kTile_j = 3;" in prologue
    assert "for (int64_t j_out = 0; j_out < B0_size; j_out += kTile_j) {" in cpp
    assert "int64_t j_out_end = std::min(j_out + kTile_j, B0_size);" in cpp
    assert "int pA1_row_end = A1_pos[pA0 + 1];" in cpp
    assert (
        "int pA1_panel_begin = (int)(std::lower_bound(A1_crd + A1_pos[pA0], "
        "A1_crd + pA1_row_end, j_out) - A1_crd);"
    ) in cpp
    assert (
        "int pA1_end = (int)(std::lower_bound(A1_crd + pA1_panel_begin, "
        "A1_crd + pA1_row_end, j_out_end) - A1_crd);"
    ) in cpp
    assert "for (int pA1 = pA1_panel_begin; pA1 < pA1_end; pA1++) {" in cpp
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(A1_pos[A0_size], "
        "A0_size)) schedule(dynamic, scorch_chunk(A0_size, A1_pos[A0_size]))"
    ) in cpp
    assert "__builtin_prefetch" in cpp
    assert "_B_val_ptr" in cpp
    # The serial origin loop carries no parallel pragma of its own.
    origin_at = cpp.index("for (int64_t j_out")
    preceding = cpp[:origin_at].rstrip().rsplit("\n", 1)[-1]
    assert "#pragma" not in preceding


def test_panel_row_loop_is_marked_after_the_managed_passes():
    """The row policy must be the legacy explicit form: exactly one
    parallel pragma, attached to the row loop inside the origin loop."""

    cpp = scheduled_panel_cpp(width=2)
    assert cpp.count("#pragma omp parallel for") == 1
    pragma_at = cpp.index("#pragma omp parallel for")
    origin_at = cpp.index("for (int64_t j_out")
    row_at = cpp.index("for (int64_t i = 0;")
    assert origin_at < pragma_at < row_at


def test_panel_row_must_sit_between_origin_and_window():
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    fixture = build_panel_spmm(width=3)
    builder = fixture.builder
    # Rebuild with the row loop OUTSIDE the panel: verifier-legal, but not
    # the migrated emission shape.
    window = builder.sparse_window_for(
        fixture.panel.tile,
        fixture.window.cursor,
        fixture.window.position,
        fixture.window.coord_index,
        fixture.window.body,
    )
    panel = builder.panel_outer_for(
        fixture.panel.tile,
        fixture.panel.index,
        fixture.panel.dimension,
        3,
        fixture.panel.bound_tensor,
        fixture.panel.bound_level,
        builder.block((window,)),
    )
    row_loop = builder.dense_for(fixture.row, fixture.dim_i, builder.block((panel,)))
    program = builder.program(
        fixture.program.dimensions,
        fixture.program.tensors,
        fixture.program.inputs,
        fixture.program.outputs,
        builder.block((row_loop,)),
    )
    from scorch.compiler.loopir.verifier import verify_program

    verify_program(program)
    shapes, result_shape = panel_spmm_shapes(fixture)
    expect_target_code("unsupported_program_shape", program, shapes, result_shape)


def test_panel_names_participate_in_generated_name_collisions():
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    for stolen in ("j_out", "j_out_end", "kTile_j"):
        fixture = build_panel_spmm(width=3)
        forge(fixture.program.dimensions[2], name=stolen)
        shapes, result_shape = panel_spmm_shapes(fixture)
        expect_target_code(
            "generated_name_collision", fixture.program, shapes, result_shape
        )


def test_panel_width_beyond_constexpr_int_fails_at_target():
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    fixture = build_panel_spmm(width=MAX_AFFINE_TILE_WIDTH + 1)
    shapes, result_shape = panel_spmm_shapes(fixture)
    expect_target_code("unsupported_tile_width", fixture.program, shapes, result_shape)


def test_panel_parallel_row_must_partition_the_dense_result():
    """A bare verified LoopProgram must not bypass the LoopPlan race gate."""

    from scorch.compiler.loopir.nodes import LevelKind
    from scorch.compiler.loopir.verifier import verify_program
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    fixture = build_panel_spmm(width=3)
    result_decl = fixture.program.tensors[2]
    forge(
        result_decl,
        dimensions=(fixture.dim_k,),
        levels=(fixture.builder.level(LevelKind.DENSE, 0),),
    )
    leaf = fixture.free_loop.body.statements[0]
    forge(
        leaf,
        indices=(fixture.builder.index_value(fixture.free),),
    )
    verify_program(fixture.program)
    expect_target_code(
        "unsupported_program_shape",
        fixture.program,
        {fixture.a: (4, 5), fixture.b: (5, 6)},
        (6,),
    )


def _panel_llir_locations(statements):
    found = []
    for index, statement in enumerate(statements):
        found.append((statements, index, statement))
        if isinstance(statement, (llir.ForLoop, llir.WhileLoop, llir.ForLoopAuto)):
            found.extend(_panel_llir_locations(statement.body))
        elif isinstance(statement, llir.IfThenElse):
            for body in (statement.then_body, statement.else_body):
                if body:
                    found.extend(_panel_llir_locations(body))
            for body in statement.then_body_list or ():
                found.extend(_panel_llir_locations(body))
    return found


def _corrupt_completed_panel_artifact(statements, corruption):
    locations = _panel_llir_locations(statements)
    loops = {
        statement.scorch_index_var: (container, index, statement)
        for container, index, statement in locations
        if isinstance(statement, llir.ForLoop)
    }
    end_location = next(
        (container, index, statement)
        for container, index, statement in locations
        if type(statement) is llir.VarInit and statement.var.name == "pA1_end"
    )
    if corruption == "row_condition":
        row = loops["i"][2]
        row.cond = llir.BinOp("<=", row.cond.left, row.cond.right)
    elif corruption == "row_update":
        loops["i"][2].update = llir.FunctionCall("advance_i", ())
    elif corruption == "inner_loop":
        inner = loops["k"][2]
        loop_var = llir.Var("q", inner.init.var.type)
        inner.init = llir.VarInit(loop_var, llir.Literal(0))
        inner.cond = llir.BinOp("<", loop_var, inner.cond.right)
        inner.update = llir.Increment(loop_var)
    elif corruption == "malformed_loop_var":
        delattr(loops["k"][2].init.var, "name")
    elif corruption == "window_end":
        end_location[2].value = llir.Literal(0, llir.DataType.INT)
    elif corruption == "window_end_after_loop":
        end_container, end_index, end_init = end_location
        window_container, _window_index, window = loops["j"]
        assert end_container is window_container
        del end_container[end_index]
        end_container.insert(end_container.index(window) + 1, end_init)
    elif corruption == "sibling_window":
        end_container, end_index, end_init = end_location
        window_container, window_index, window = loops["j"]
        assert end_container is window_container
        for index in sorted((end_index, window_index), reverse=True):
            del end_container[index]
        statements.extend((end_init, window))
    elif corruption == "extra_loop":
        loop_var = llir.Var("q", llir.DataType.INT64)
        statements.append(
            llir.ForLoop(
                llir.VarInit(loop_var, llir.Literal(0)),
                llir.BinOp("<", loop_var, llir.Literal(1)),
                llir.Increment(loop_var),
                [],
            )
        )
    elif corruption == "unknown_statement":

        class UnknownStatement(llir.Stmt):
            pass

        statements.append(UnknownStatement())
    elif corruption == "duplicate_window_end":
        end_location[0].insert(end_location[1], copy.deepcopy(end_location[2]))
    elif corruption == "malformed_neighbor_init":
        malformed = copy.deepcopy(end_location[2])
        delattr(malformed, "var")
        end_location[0].insert(end_location[1], malformed)
    elif corruption == "malformed_neighbor_var":
        malformed = copy.deepcopy(end_location[2])
        delattr(malformed.var, "name")
        end_location[0].insert(end_location[1], malformed)
    elif corruption == "malformed_window_value":
        assert type(end_location[2].value) is llir.ArrayAccess
        object.__delattr__(end_location[2].value, "array")
    elif corruption == "window_value_subclass":

        class ExplodingExpr(llir.Expr):
            def __eq__(self, _other):
                raise RuntimeError("forged equality must never run")

        end_location[2].value = ExplodingExpr()
    elif corruption == "missing_optional_header":
        delattr(loops["k"][2], "before_parallel_body")
    elif corruption == "atomic_marker":
        setattr(loops["k"][2], "_use_atomic_scheduling", True)
    elif corruption == "cyclic_loop_body":
        row = loops["i"][2]
        row.body = [row]
    elif corruption == "shared_loop":
        loops["i"][2].body.append(loops["k"][2])
    else:
        raise AssertionError(f"unknown test corruption {corruption!r}")


def _install_panel_completion_corruption(monkeypatch, corruption):
    original = LLIRPassManager.run_production_pipeline

    def corrupt_after_managed_passes(manager, *args, **kwargs):
        result = original(manager, *args, **kwargs)
        _corrupt_completed_panel_artifact(result.artifact.value, corruption)
        return result

    monkeypatch.setattr(
        LLIRPassManager,
        "run_production_pipeline",
        corrupt_after_managed_passes,
    )


@pytest.mark.parametrize(
    "corruption",
    (
        "row_condition",
        "row_update",
        "inner_loop",
        "malformed_loop_var",
        "window_end",
        "window_end_after_loop",
        "sibling_window",
        "extra_loop",
        "unknown_statement",
        "duplicate_window_end",
        "malformed_neighbor_init",
        "malformed_neighbor_var",
        "malformed_window_value",
        "window_value_subclass",
        "missing_optional_header",
        "atomic_marker",
        "cyclic_loop_body",
        "shared_loop",
    ),
)
def test_panel_completion_fails_closed_on_post_pass_corruption(monkeypatch, corruption):
    _install_panel_completion_corruption(monkeypatch, corruption)
    with pytest.raises(LoopIRTargetError) as error:
        scheduled_panel_cpp(width=3)
    assert error.value.defect.code == "panel_completion_lost"


def test_panel_completion_requires_the_parallel_marker_to_take_effect(monkeypatch):
    from scorch.compiler.loopir import lower_llir as loopir_lower_llir

    monkeypatch.setattr(
        loopir_lower_llir,
        "mark_first_for_loop_parallel",
        lambda _statements, _cluster: None,
    )
    with pytest.raises(LoopIRTargetError) as error:
        scheduled_panel_cpp(width=3)
    assert error.value.defect.code == "panel_completion_lost"


@pytest.mark.parametrize(
    "field,value",
    (
        ("omp_num_threads", "corrupt_policy()"),
        ("omp_chunk_expr", "corrupt_chunk()"),
        ("pre_parallel_body", [llir.RawStmt("corrupt();")]),
        ("_use_atomic_scheduling", True),
    ),
)
def test_panel_completion_requires_the_exact_parallel_policy(monkeypatch, field, value):
    from scorch.compiler.loopir import lower_llir as loopir_lower_llir

    original = loopir_lower_llir.mark_first_for_loop_parallel

    def corrupt_marker(statements, cluster):
        original(statements, cluster)
        setattr(statements[0], field, value)

    monkeypatch.setattr(
        loopir_lower_llir,
        "mark_first_for_loop_parallel",
        corrupt_marker,
    )
    with pytest.raises(LoopIRTargetError) as error:
        scheduled_panel_cpp(width=3)
    assert error.value.defect.code == "panel_completion_lost"


def test_panel_completion_revalidates_the_owned_work_fact(monkeypatch):
    from scorch.compiler.loopir import lower_llir as loopir_lower_llir

    from tests.test_scorch.test_loopir_verifier import forge

    original = loopir_lower_llir._TargetLowering.complete_panel

    def mutate_work(self, function):
        assert self.parallel is not None
        forge(self.parallel.work, nnz=None)
        return original(self, function)

    monkeypatch.setattr(
        loopir_lower_llir._TargetLowering,
        "complete_panel",
        mutate_work,
    )
    with pytest.raises(LoopIRTargetError) as error:
        scheduled_panel_cpp(width=3)
    assert error.value.defect.code == "panel_completion_lost"


def test_panel_completion_failure_owns_stage_and_keeps_completed_pass_records(
    monkeypatch,
):
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    _install_panel_completion_corruption(monkeypatch, "extra_loop")
    fixture = build_panel_spmm(width=3)
    shapes, result_shape = panel_spmm_shapes(fixture)
    options = CompileOptions.from_environment()
    context = CompilationContext(options)
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            fixture.program,
            input_shapes=shapes,
            result_shape=result_shape,
            compile_options=options,
            compilation_context=context,
        )
    assert error.value.defect.code == "panel_completion_lost"
    assert context.stage_run_records == ()
    assert [record.pass_name for record in context.llir_pass_run_records] == [
        "insert_sparse_prefetch",
        "hoist_dense_pointers",
        "eliminate_single_iteration_loops",
        "hoist_loop_invariant_factors",
        "rewrite_dynamic_vector_accesses",
    ]
    with pytest.raises(CompilationContextError) as terminal:
        context.begin_stage(
            CompilerStageId.LLIR_TO_CPP_GENERATION,
            compile_options=options,
        )
    assert terminal.value.diagnostic.code == "failed_compilation"


# -- staged-relayout target boundaries ---------------------------------------


def relayout_program(scope=None, width=3, strip=4):
    from scorch.compiler.loopir.nodes import RelayoutScope

    from tests.test_scorch.test_loopir_verifier import build_relayout_spmm

    return build_relayout_spmm(
        scope if scope is not None else RelayoutScope.PANEL,
        width=width,
        strip=strip,
    )


def relayout_shapes(fixture):
    return {fixture.a: (4, 5), fixture.b: (5, 6)}


def relayout_llir_nodes(root):
    """Walk one ordinary test-owned LLIR tree without invoking equality."""

    pending = [root]
    seen = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        yield value
        if isinstance(value, llir.Node):
            pending.extend(
                child
                for child in vars(value).values()
                if isinstance(child, (llir.Node, list, tuple))
            )
        elif isinstance(value, (list, tuple)):
            pending.extend(value)


def test_relayout_target_emits_the_legacy_packed_source():
    """Direct structural activation on a bare verified LoopIR program."""

    from scorch.compiler.loopir.nodes import RelayoutScope

    fixture = relayout_program(RelayoutScope.PANEL)
    lowered = lower_loopir_to_llir(
        fixture.program,
        input_shapes=relayout_shapes(fixture),
        result_shape=(4, 6),
    )
    from scorch.compiler.codegen import LLIRLowerer

    source = LLIRLowerer().lower_llir(lowered)
    assert (
        "std::vector<float> packed_B_storage((size_t)kTile_j * (size_t)kTile_k);"
        in source
    )
    assert "float* __restrict__ packed_B = packed_B_storage.data();" in source
    assert "// Pack B j panel into contiguous j-major storage" in source
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads((j_out_end - "
        "j_out) * kTile_k, (j_out_end - j_out))) schedule(static)" in source
    )
    assert (
        "packed_B[(j_pack - j_out) * kTile_k + k_pack] = "
        "B_val[j_pack * B1_size + k_packed];"
    ) in source
    assert "if (j < j_out || j >= j_out_end) {" in source
    assert "packed_B[(j - j_out) * kTile_k + k_in];" in source
    assert (
        "__builtin_prefetch(&packed_B[(A1_crd[pA1 + 1] - j_out) * kTile_k], 0, 1)"
        in source
    )
    # The direct operand read is fully redirected.
    assert "B_val[pB1]" not in source

    fixture = relayout_program(RelayoutScope.PACK_AXIS)
    lowered = lower_loopir_to_llir(
        fixture.program,
        input_shapes=relayout_shapes(fixture),
        result_shape=(4, 6),
    )
    source = LLIRLowerer().lower_llir(lowered)
    assert (
        "std::vector<float> packed_B_storage((size_t)B0_size * (size_t)kTile_k);"
        in source
    )
    assert "// Pack B full j axis into contiguous j-major storage" in source
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(B0_size * "
        "kTile_k, B0_size)) schedule(static)" in source
    )
    assert "packed_B[j_pack * kTile_k + k_pack] = " in source
    assert "packed_B[j * kTile_k + k_in];" in source
    assert "__builtin_prefetch(&packed_B[A1_crd[pA1 + 1] * kTile_k], 0, 1)" in source
    assert "B_val[pB1]" not in source


def test_relayout_target_requires_the_exact_chain_shape():
    from tests.test_scorch.test_loopir_verifier import forge

    # A second staging region is outside the family.  Both regions are
    # read so the program stays verifier-legal; the target's collect gate
    # rejects the second region before validation.
    from scorch.compiler.loopir.nodes import BinaryOp as LoopIRBinaryOp

    fixture = relayout_program()
    inner_decl = fixture.builder.relayout_decl(
        fixture.builder.new_relayout_id(),
        fixture.b,
        fixture.panel_tile,
        fixture.pack_tile,
        fixture.decl.scope,
    )
    inner_stage = fixture.builder.relayout_stage(inner_decl, fixture.stage.body)
    inner_read = fixture.builder.staged_read(
        inner_decl.relayout,
        (
            fixture.builder.index_value(fixture.col),
            fixture.builder.index_value(fixture.free),
        ),
    )
    forge(
        fixture.leaf,
        value=fixture.builder.binary(
            LoopIRBinaryOp.MUL, fixture.leaf.value, inner_read
        ),
    )
    forge(fixture.stage, body=fixture.builder.block((inner_stage,)))
    expect_target_code(
        "unsupported_program_shape",
        fixture.program,
        relayout_shapes(fixture),
        (4, 6),
    )

    # A verifier-legal PANEL-scope region nested inside the row loop is
    # outside the audited placement: the region must open directly below
    # its scope loop.  (A region naming a foreign pack or panel identity
    # cannot verify at all — the verifier's relayout_scope_mismatch owns
    # that boundary.)
    fixture = relayout_program()
    nested_stage = fixture.builder.relayout_stage(fixture.decl, fixture.row_loop.body)
    nested_row = fixture.builder.dense_for(
        fixture.row, fixture.dim_i, fixture.builder.block((nested_stage,))
    )
    forge(fixture.panel, body=fixture.builder.block((nested_row,)))
    expect_target_code(
        "unsupported_program_shape",
        fixture.program,
        relayout_shapes(fixture),
        (4, 6),
    )

    # A region at the wrong depth for its scope.
    from scorch.compiler.loopir.nodes import RelayoutScope

    fixture = relayout_program(RelayoutScope.PACK_AXIS)
    forge(fixture.decl, scope=RelayoutScope.PANEL)
    # Verifier-level scope discipline fires first; bypass it by forging
    # a PANEL region at the PACK_AXIS position is verifier-invalid, so
    # this boundary is unreachable through verified programs — the
    # verifier's relayout_scope_mismatch owns it.
    with pytest.raises(LoopIRVerificationError):
        lower_loopir_to_llir(
            fixture.program,
            input_shapes=relayout_shapes(fixture),
            result_shape=(4, 6),
        )


def test_relayout_target_requires_the_panel_family():
    """A staging region without its panel pair cannot reach emission."""

    from tests.test_scorch.test_loopir_verifier import forge

    fixture = relayout_program()
    # Erase the panel pair from the region body: the window becomes a
    # plain sparse loop and the panel origin disappears; the staged read
    # then has no window coordinate, so verification fails closed before
    # the target boundary is consulted.
    forge(fixture.decl, panel=fixture.builder.new_tile_id())
    with pytest.raises(LoopIRVerificationError):
        lower_loopir_to_llir(
            fixture.program,
            input_shapes=relayout_shapes(fixture),
            result_shape=(4, 6),
        )


def test_relayout_completion_requires_recorded_state(monkeypatch):
    """A lost panel-completion record fails closed, never guesses."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = relayout_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def forgetful(self, function):
        completed = original(self, function)
        self._panel_completion = None
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", forgetful)
    expect_target_code(
        "relayout_completion_lost",
        fixture.program,
        relayout_shapes(fixture),
        (4, 6),
    )


def test_relayout_completion_requires_the_window_coordinate(monkeypatch):
    """A corrupted resolved-coordinate declaration fails closed."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = relayout_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def corrupting(self, function):
        completed = original(self, function)
        _pack, _panel, _row, window = self._panel_completion
        for statement in window.body:
            if type(statement) is llir.VarInit and statement.var.name == "j":
                statement.var = llir.Var("j_forged", statement.var.type)
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", corrupting)
    expect_target_code(
        "relayout_completion_lost",
        fixture.program,
        relayout_shapes(fixture),
        (4, 6),
    )


def test_relayout_completion_rejects_relocated_window_coordinate(monkeypatch):
    """An unchanged declaration moved below its uses is not re-identified."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = relayout_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def relocating(self, function):
        completed = original(self, function)
        _pack, _panel, _row, window = self._panel_completion
        index, coordinate = next(
            (index, statement)
            for index, statement in enumerate(window.body)
            if type(statement) is llir.VarInit and statement.var.name == "j"
        )
        del window.body[index]
        window.body.append(coordinate)
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", relocating)
    expect_target_code(
        "relayout_completion_lost",
        fixture.program,
        relayout_shapes(fixture),
        (4, 6),
    )


def test_relayout_completion_rejects_relocated_coordinate_context(monkeypatch):
    """Moving the intact context below its uses does not preserve dominance."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = relayout_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def relocating(self, function):
        completed = original(self, function)
        _pack, _panel, _row, window = self._panel_completion
        coordinate_index = next(
            index
            for index, statement in enumerate(window.body)
            if type(statement) is llir.VarInit and statement.var.name == "j"
        )
        context = window.body[coordinate_index - 1 : coordinate_index + 2]
        assert [type(statement) for statement in context] == [
            llir.Comment,
            llir.VarInit,
            llir.BlankLine,
        ]
        del window.body[coordinate_index - 1 : coordinate_index + 2]
        window.body.extend(context)
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", relocating)
    expect_target_code(
        "relayout_completion_lost",
        fixture.program,
        relayout_shapes(fixture),
        (4, 6),
    )


def test_relayout_completion_anchors_metadata_to_the_physical_access(monkeypatch):
    """Valid metadata moved to the wrong operand cannot redirect that operand."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = relayout_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def swapping(self, function):
        completed = original(self, function)
        _pack, _panel, row, _window = self._panel_completion
        accesses = [
            node
            for node in relayout_llir_nodes(row.body)
            if isinstance(node, llir.Expr)
            and type(getattr(node, "tensor_access", None)) is llir.TensorAccessMetadata
            and node.tensor_access.role is llir.TensorAccessRole.INPUT_READ
        ]
        staged = next(
            node
            for node in accesses
            if node.tensor_access.tensor_id == self.relayout.operand
        )
        other = next(
            node
            for node in accesses
            if node.tensor_access.tensor_id != self.relayout.operand
        )
        staged_metadata = staged.tensor_access
        other_metadata = other.tensor_access
        object.__setattr__(staged, "tensor_access", other_metadata)
        object.__setattr__(other, "tensor_access", staged_metadata)
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", swapping)
    expect_target_code(
        "relayout_completion_lost",
        fixture.program,
        relayout_shapes(fixture),
        (4, 6),
    )


def test_relayout_completion_owns_a_detached_metadata_fingerprint(monkeypatch):
    """In-place provenance mutation cannot mutate the retained fingerprint."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = relayout_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def mutating(self, function):
        completed = original(self, function)
        _pack, _panel, row, _window = self._panel_completion
        staged = next(
            node
            for node in relayout_llir_nodes(row.body)
            if isinstance(node, llir.Expr)
            and type(getattr(node, "tensor_access", None)) is llir.TensorAccessMetadata
            and node.tensor_access.tensor_id == self.relayout.operand
        )
        metadata = staged.tensor_access
        assert type(metadata) is llir.TensorAccessMetadata
        object.__setattr__(
            metadata.access_id,
            "value",
            metadata.access_id.value + 1,
        )
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", mutating)
    expect_target_code(
        "relayout_completion_lost",
        fixture.program,
        relayout_shapes(fixture),
        (4, 6),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_type",
        "missing_field",
        "hostile_identity",
        "hostile_extra_key",
        "cyclic",
    ],
)
def test_relayout_completion_owns_malformed_access_state(monkeypatch, mutation):
    """Malformed post-pass access state stays inside the target diagnostic."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = relayout_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def corrupting(self, function):
        completed = original(self, function)
        _pack, _panel, row, _window = self._panel_completion
        staged = next(
            node
            for node in relayout_llir_nodes(row.body)
            if isinstance(node, llir.Expr)
            and type(getattr(node, "tensor_access", None)) is llir.TensorAccessMetadata
            and node.tensor_access.tensor_id == self.relayout.operand
        )
        if mutation == "wrong_type":
            object.__setattr__(staged, "tensor_access", object())
        elif mutation == "missing_field":
            metadata = staged.tensor_access
            assert type(metadata) is llir.TensorAccessMetadata
            object.__delattr__(metadata, "tensor_id")
        elif mutation == "hostile_identity":
            metadata = staged.tensor_access
            assert type(metadata) is llir.TensorAccessMetadata
            hostile = SymbolId(0)
            object.__setattr__(hostile, "value", object())
            object.__setattr__(metadata, "tensor_id", hostile)
        elif mutation == "hostile_extra_key":

            class HostileKey:
                def __init__(self):
                    self.armed = False

                def __hash__(self):
                    return hash("array")

                def __eq__(self, other):
                    if self.armed:
                        raise RuntimeError("hostile access-key equality")
                    return False

            key = HostileKey()
            object.__getattribute__(staged, "__dict__")[key] = None
            key.armed = True
        else:
            assert type(staged) is llir.ArrayAccess
            object.__setattr__(staged, "index", staged)
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", corrupting)
    expect_target_code(
        "relayout_completion_lost",
        fixture.program,
        relayout_shapes(fixture),
        (4, 6),
    )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "noncanonical_duplicate"])
def test_relayout_completion_requires_one_sparse_prefetch(monkeypatch, mutation):
    """The typed completion never silently drops or deduplicates its prefetch."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = relayout_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def mutating(self, function):
        completed = original(self, function)
        _pack, _panel, _row, window = self._panel_completion
        guards = [
            statement
            for statement in window.body
            if type(statement) is llir.GuardedCallStmt
        ]
        assert len(guards) == 1
        if mutation == "missing":
            window.body.remove(guards[0])
        elif mutation == "duplicate":
            window.body.insert(0, copy.deepcopy(guards[0]))
        else:
            decoy = copy.deepcopy(guards[0])
            call = decoy.call
            object.__setattr__(
                call,
                "args",
                (*call.args[:-1], llir.Literal(2, llir.DataType.INT)),
            )
            nested = next(
                node
                for node in relayout_llir_nodes(window.body)
                if type(node) is llir.ForLoop
            )
            nested.body.append(decoy)
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", mutating)
    expect_target_code(
        "relayout_completion_lost",
        fixture.program,
        relayout_shapes(fixture),
        (4, 6),
    )


# -- heap result-tile target boundary -----------------------------------------


def heap_program(strip=3, dtype=None):
    from scorch.compiler.loopir.nodes import ScalarType as _ScalarType

    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    return build_heap_spmm(
        strip=strip,
        dtype=dtype if dtype is not None else _ScalarType.FLOAT32,
    )


def heap_shapes(fixture):
    return {fixture.a: (4, 5), fixture.b: (5, 6)}


def test_heap_target_emits_the_legacy_compact_source():
    """Direct structural activation on a bare verified LoopIR program."""

    fixture = heap_program()
    lowered = lower_loopir_to_llir(
        fixture.program,
        input_shapes=heap_shapes(fixture),
        result_shape=(4, 6),
    )
    from scorch.compiler.codegen import LLIRLowerer

    source = LLIRLowerer().lower_llir(lowered)
    assert (
        "std::vector<float> tiled_C_storage((size_t)C0_size * (size_t)kTile_k);"
        in source
    )
    assert "float* __restrict__ tiled_C = tiled_C_storage.data();" in source
    assert "// Initialize compact result tile for C" in source
    assert "// Copy compact result tile to C" in source
    assert "tiled_C[C_tile_init * kTile_k + k_tile_init] = 0.0f;" in source
    assert (
        "C_values[C_tile_copy * C1_size + k_copy_logical] = "
        "tiled_C[C_tile_copy * kTile_k + k_tile_copy];"
    ) in source
    assert "tiled_C[pC0 * kTile_k + k_in] += A_val[pA1] * B_val[pB1];" in source
    assert "scorch_zero_dense(C_values" not in source
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(A1_pos[A0_size], "
        "A0_size)) schedule(dynamic, scorch_chunk(A0_size, A1_pos[A0_size]))"
    ) in source


def test_heap_target_rejects_post_construction_semantic_leaf_replacement(monkeypatch):
    """The actual TiledReduce cannot diverge from its synthetic direct view."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module
    from scorch.compiler.loopir.verifier import verify_program

    fixture = heap_program()
    original = lower_llir_module._TargetLowering.raw_loop_statements

    def replacing(self):
        assert self._tiled_leaf is not None
        value = self._tiled_leaf.value
        assert type(value) is lower_llir_module.BinaryExpr
        object.__setattr__(self._tiled_leaf, "value", value.lhs)
        verify_program(self.program)
        return original(self)

    monkeypatch.setattr(
        lower_llir_module._TargetLowering,
        "raw_loop_statements",
        replacing,
    )
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            fixture.program,
            input_shapes=heap_shapes(fixture),
            result_shape=(4, 6),
        )
    assert error.value.defect.code == "unsupported_program_shape"
    assert "owning statement" in error.value.defect.message


@pytest.mark.parametrize(
    "mutation",
    ["region_body", "leaf_slot", "result_tile_decl"],
    ids=str,
)
def test_heap_target_binds_region_body_leaf_and_declaration(mutation):
    """Heap completion cannot emit through stale retained region objects."""

    from scorch.compiler.loopir.build import LoopIRBuilder

    fixture = heap_program()

    def mutate(target):
        if mutation == "region_body":
            object.__setattr__(
                fixture.region,
                "body",
                copy.deepcopy(fixture.region.body),
            )
            return
        if mutation == "leaf_slot":
            object.__setattr__(
                fixture.pack_point.body,
                "statements",
                (copy.deepcopy(fixture.leaf),),
            )
            return
        old = fixture.region.decl
        builder = LoopIRBuilder.resuming(fixture.program)
        fresh = builder.result_tile_decl(
            old.result_tile,
            old.result,
            old.pack,
        )
        object.__setattr__(fixture.region, "decl", fresh)

    _expect_post_construction_graph_rejection(
        fixture.program,
        heap_shapes(fixture),
        (4, 6),
        mutate,
    )


@pytest.mark.parametrize("mutation", ["retarget", "missing"], ids=str)
def test_heap_target_rejects_post_construction_index_mutation(monkeypatch, mutation):
    """Shared index children cannot redirect a compact write or raw-escape."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = heap_program()
    original = lower_llir_module._TargetLowering.raw_loop_statements

    def mutating(self):
        assert self._tiled_leaf is not None
        indices = self._tiled_leaf.indices
        if mutation == "retarget":
            object.__setattr__(indices[0], "index", indices[1].index)
        else:
            object.__delattr__(indices[0], "index")
        return original(self)

    monkeypatch.setattr(
        lower_llir_module._TargetLowering,
        "raw_loop_statements",
        mutating,
    )
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            fixture.program,
            input_shapes=heap_shapes(fixture),
            result_shape=(4, 6),
        )
    assert error.value.defect.code == "unsupported_program_shape"


def test_heap_generated_names_reserve_flattened_container_declarations():
    """Name allocation uses every declaration in the emitted C++ scope."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module
    from scorch.compiler.schedule_lowerer import _heap_result_tile_names

    # Codegen flattens nested statement containers without adding a scope.
    function = llir.Function(
        llir.DataType.VOID,
        "evaluate",
        (),
        [
            [
                llir.VarInit(
                    llir.Var("tiled_C", llir.DataType.INT),
                    llir.Literal(0),
                )
            ]
        ],
    )
    reserved_names = lower_llir_module._validate_result_tile_rendered_text(
        function,
        protected_names=set(),
    )
    names = _heap_result_tile_names(
        function,
        "C",
        "k",
        reserved_names=reserved_names,
    )
    assert names[0] == "tiled_C_1"


@pytest.mark.parametrize("declaration_owner", ["split_pre_body", "atomic_counter"])
def test_heap_generated_names_reserve_codegen_declarations(declaration_owner):
    """Auxiliary OpenMP declarations participate in compact-name allocation."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module
    from scorch.compiler.schedule_lowerer import _heap_result_tile_names

    chunk = llir.Var("chunk", llir.DataType.INT)
    bound = llir.Var("bound", llir.DataType.INT)
    loop = llir.ForLoop(
        init=llir.VarInit(
            llir.Var("i", llir.DataType.INT),
            llir.Literal(0),
        ),
        cond=llir.BinOp(
            "<",
            llir.Var("i", llir.DataType.INT),
            bound,
        ),
        update=llir.Increment(llir.Var("i", llir.DataType.INT)),
        body=[llir.Continue()],
    )
    if declaration_owner == "split_pre_body":
        loop.omp_parallel_for = True
        loop.pre_parallel_body = [
            llir.VarInit(
                llir.Var("tiled_C", llir.DataType.INT),
                llir.Literal(0),
            )
        ]
    else:
        loop._use_atomic_scheduling = True
        loop._atomic_counter_var = "tiled_C"
        loop._atomic_chunk_var = "chunk"
        # Codegen declares _start before consulting the loop bound.
        loop._loop_bound = "_start"
        loop.omp_num_threads = "scorch_nthreads(tiled_C, bound)"
    function = llir.Function(
        llir.DataType.VOID,
        "evaluate",
        (chunk, bound),
        [loop],
    )
    reserved_names = lower_llir_module._validate_result_tile_rendered_text(
        function,
        protected_names=set(),
    )
    names = _heap_result_tile_names(
        function,
        "C",
        "k",
        reserved_names=reserved_names,
    )
    assert names[0] == "tiled_C_1"


def test_heap_completion_anchors_metadata_to_the_physical_write(monkeypatch):
    """Valid metadata moved to the wrong access cannot redirect that access."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = heap_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def swapping(self, function):
        completed = original(self, function)
        accesses = [
            node
            for node in relayout_llir_nodes(function.body)
            if isinstance(node, llir.Expr)
            and type(getattr(node, "tensor_access", None)) is llir.TensorAccessMetadata
        ]
        write = next(
            node
            for node in accesses
            if node.tensor_access.role is llir.TensorAccessRole.RESULT_WRITE
        )
        other = next(
            node
            for node in accesses
            if node.tensor_access.role is llir.TensorAccessRole.INPUT_READ
        )
        write_metadata = write.tensor_access
        other_metadata = other.tensor_access
        object.__setattr__(write, "tensor_access", other_metadata)
        object.__setattr__(other, "tensor_access", write_metadata)
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", swapping)
    expect_target_code(
        "result_tile_completion_lost",
        fixture.program,
        heap_shapes(fixture),
        (4, 6),
    )


def test_heap_completion_owns_a_detached_metadata_fingerprint(monkeypatch):
    """In-place provenance mutation cannot mutate the retained fingerprint."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = heap_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def mutating(self, function):
        completed = original(self, function)
        write = next(
            node
            for node in relayout_llir_nodes(function.body)
            if isinstance(node, llir.Expr)
            and type(getattr(node, "tensor_access", None)) is llir.TensorAccessMetadata
            and node.tensor_access.role is llir.TensorAccessRole.RESULT_WRITE
        )
        metadata = write.tensor_access
        object.__setattr__(
            metadata.access_id,
            "value",
            metadata.access_id.value + 1,
        )
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", mutating)
    expect_target_code(
        "result_tile_completion_lost",
        fixture.program,
        heap_shapes(fixture),
        (4, 6),
    )


@pytest.mark.parametrize(
    "mutation",
    ["wrong_type", "missing_field", "hostile_identity", "cyclic"],
)
def test_heap_completion_owns_malformed_write_state(monkeypatch, mutation):
    """Malformed post-pass write state stays inside the target diagnostic."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = heap_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def corrupting(self, function):
        completed = original(self, function)
        write = next(
            node
            for node in relayout_llir_nodes(function.body)
            if isinstance(node, llir.Expr)
            and type(getattr(node, "tensor_access", None)) is llir.TensorAccessMetadata
            and node.tensor_access.role is llir.TensorAccessRole.RESULT_WRITE
        )
        if mutation == "wrong_type":
            object.__setattr__(write, "tensor_access", object())
        elif mutation == "missing_field":
            metadata = write.tensor_access
            object.__delattr__(metadata, "tensor_id")
        elif mutation == "hostile_identity":
            metadata = write.tensor_access
            hostile = SymbolId(0)
            object.__setattr__(hostile, "value", object())
            object.__setattr__(metadata, "tensor_id", hostile)
        else:
            assert type(write) is llir.ArrayAccess
            object.__setattr__(write, "index", write)
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", corrupting)
    expect_target_code(
        "result_tile_completion_lost",
        fixture.program,
        heap_shapes(fixture),
        (4, 6),
    )


_HEAP_REVIEW_CONTROL_AND_EFFECT_MUTATIONS = {
    "top_level_break",
    "top_level_continue",
    "missing_conditional_condition",
    "empty_conditional_then",
    "missing_conditional_branches",
    "mismatched_conditional_branches",
    "mutate_result_extent",
    "mutate_result_position",
    "mutate_result_shape",
    "move_result_shape",
    "escape_result_shape_address",
    "unowned_result_pointer_call",
    "guarded_call_protected_argument",
    "member_call_protected_argument",
    "member_call_expression_protected_argument",
    "forged_result_shape_validation",
    "duplicate_result_shape_validation",
    "late_result_shape_validation",
    "late_input_validation",
    "extra_protected_torch_empty",
}


def _apply_heap_review_control_or_effect_mutation(
    lowering,
    function,
    owner,
    final_assembly,
    mutation,
):
    """Inject one review-only malformed control/effect boundary."""

    if mutation == "top_level_break":
        function.body.insert(final_assembly, llir.Break())
    elif mutation == "top_level_continue":
        function.body.insert(final_assembly, llir.Continue())
    elif mutation == "missing_conditional_condition":
        function.body.insert(final_assembly, llir.IfThenElse())
    elif mutation == "empty_conditional_then":
        function.body.insert(
            final_assembly,
            llir.IfThenElse(
                cond=llir.Literal(True),
                then_body=[],
            ),
        )
    elif mutation == "missing_conditional_branches":
        function.body.insert(
            final_assembly,
            llir.IfThenElse(
                cond_list=[llir.Literal(True)],
            ),
        )
    elif mutation == "mismatched_conditional_branches":
        function.body.insert(
            final_assembly,
            llir.IfThenElse(
                cond_list=[
                    llir.Literal(True),
                    llir.Literal(False),
                ],
                then_body_list=[[llir.BlankLine()]],
            ),
        )
    elif mutation == "mutate_result_extent":
        extent_declaration = next(
            index
            for index, statement in enumerate(function.body)
            if type(statement) is llir.VarInit and statement.var.name == "C0_size"
        )
        function.body.insert(
            extent_declaration + 1,
            llir.Assign(
                llir.Var("C0_size", llir.DataType.INT64),
                llir.Literal(0, llir.DataType.INT64),
            ),
        )
    elif mutation == "mutate_result_position":
        located = lowering._locate_statement(function.body, owner)
        assert located is not None
        body, position = located
        body.insert(
            position,
            llir.Increment(llir.Var("pC1", llir.DataType.INT)),
        )
    elif mutation == "mutate_result_shape":
        function.body.insert(
            1,
            llir.MemberCallStmt(
                base=llir.Var(
                    "result_shape",
                    llir.DataType.STD_VECTOR_INT,
                ),
                member="clear",
            ),
        )
    elif mutation == "move_result_shape":
        function.body.insert(
            1,
            llir.VarInit(
                llir.Var("review_shape", llir.DataType.STD_VECTOR_INT),
                llir.FunctionCall(
                    "std::move",
                    (
                        llir.Var(
                            "result_shape",
                            llir.DataType.STD_VECTOR_INT,
                        ),
                    ),
                ),
            ),
        )
    elif mutation == "escape_result_shape_address":
        function.body.insert(
            final_assembly,
            llir.FunctionCallStmt(
                "review_mutate",
                (
                    llir.AddressOf(
                        llir.Var(
                            "result_shape",
                            llir.DataType.STD_VECTOR_INT,
                        )
                    ),
                ),
            ),
        )
    elif mutation == "guarded_call_protected_argument":
        # A guarded single-line call is an unknown callee exactly like a free
        # call statement; protected state may not ride its argument grammar.
        function.body.insert(
            final_assembly,
            llir.GuardedCallStmt(
                cond=llir.BinOp(
                    "<",
                    llir.Literal(0, llir.DataType.INT64),
                    llir.Literal(1, llir.DataType.INT64),
                ),
                call=llir.FunctionCallStmt(
                    "review_probe",
                    (llir.Var("C_capacity", llir.DataType.INT64),),
                ),
            ),
        )
    elif mutation == "member_call_protected_argument":
        function.body[final_assembly:final_assembly] = [
            llir.VarDecl(llir.Var("review_sink_vec", llir.DataType.STD_VECTOR_INT)),
            llir.MemberCallStmt(
                base=llir.Var("review_sink_vec", llir.DataType.STD_VECTOR_INT),
                member="push_back",
                args=(llir.Var("C_capacity", llir.DataType.INT64),),
            ),
        ]
    elif mutation == "member_call_expression_protected_argument":
        function.body[final_assembly:final_assembly] = [
            llir.VarDecl(llir.Var("review_sink_vec", llir.DataType.STD_VECTOR_INT)),
            llir.VarInit(
                llir.Var("review_sink", llir.DataType.INT64),
                llir.MemberCall(
                    base=llir.Var("review_sink_vec", llir.DataType.STD_VECTOR_INT),
                    member="take",
                    args=(llir.Var("C_capacity", llir.DataType.INT64),),
                ),
            ),
        ]
    elif mutation == "forged_result_shape_validation":
        # A wrong-arity forged validation would ride the binding validator's
        # name allowlist into non-compiling C++ without the census pin.
        function.body.insert(
            final_assembly,
            llir.FunctionCallStmt(
                "scorch_native::validate_jit_result_shape",
                (llir.Var("result_shape", llir.DataType.STD_VECTOR_INT),),
            ),
        )
    elif mutation == "duplicate_result_shape_validation":
        # An exact second copy of the canonical validation compiles and
        # changes behavior only through its count; the census must reject it.
        canonical = next(
            statement
            for statement in function.body
            if type(statement) is llir.FunctionCallStmt
            and statement.name == "scorch_native::validate_jit_result_shape"
        )
        function.body.insert(final_assembly, copy.deepcopy(canonical))
    elif mutation == "late_result_shape_validation":
        validation_index = next(
            index
            for index, statement in enumerate(function.body)
            if type(statement) is llir.FunctionCallStmt
            and statement.name == "scorch_native::validate_jit_result_shape"
        )
        validation = function.body.pop(validation_index)
        adjusted_final = final_assembly - int(validation_index < final_assembly)
        function.body.insert(adjusted_final, validation)
    elif mutation == "late_input_validation":
        validation_index = next(
            index
            for index, statement in enumerate(function.body)
            if type(statement) is llir.FunctionCallStmt
            and statement.name == "scorch_native::validate_jit_tensor"
        )
        validation = function.body.pop(validation_index)
        adjusted_final = final_assembly - int(validation_index < final_assembly)
        function.body.insert(adjusted_final, validation)
    elif mutation == "extra_protected_torch_empty":
        function.body.insert(
            final_assembly,
            llir.VarInit(
                llir.Var("review_tensor", llir.DataType.TORCH_TENSOR),
                llir.FunctionCall(
                    "torch::empty",
                    (
                        llir.Var(
                            "result_shape",
                            llir.DataType.STD_VECTOR_INT,
                        ),
                        llir.Var(
                            "result_shape",
                            llir.DataType.STD_VECTOR_INT,
                        ),
                    ),
                ),
            ),
        )
    else:
        assert mutation == "unowned_result_pointer_call"
        function.body.insert(
            final_assembly,
            llir.FunctionCallStmt(
                "memset",
                (
                    llir.Var(
                        "C_values",
                        llir.DataType.PTR_FLOAT32,
                    ),
                    llir.Literal(0),
                    llir.Var("C_capacity", llir.DataType.INT64),
                ),
            ),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_assignment_op",
        "unowned_physical_write",
        "unowned_top_level_write",
        "opaque_physical_write",
        "structured_tensor_alias",
        "structured_tensor_mutation",
        "opaque_call_name",
        "opaque_call_expression_name",
        "opaque_initializer_operator",
        "opaque_variable_name",
        "opaque_variable_type",
        "opaque_literal_text",
        "multiline_comment",
        "continued_comment",
        "initializer_operator_subclass",
        "unary_operator_subclass",
        "opaque_binary_operator",
        "binary_operator_subclass",
        "duplicate_tile_width_declaration",
        "duplicate_result_stack_declaration",
        "undeclared_variable",
        "use_before_declaration",
        "branch_local_variable_escape",
        "nested_function",
        "top_level_break",
        "top_level_continue",
        "missing_conditional_condition",
        "empty_conditional_then",
        "missing_conditional_branches",
        "mismatched_conditional_branches",
        "mutate_result_extent",
        "mutate_result_position",
        "mutate_result_shape",
        "move_result_shape",
        "escape_result_shape_address",
        "unowned_result_pointer_call",
        "guarded_call_protected_argument",
        "member_call_protected_argument",
        "member_call_expression_protected_argument",
        "forged_result_shape_validation",
        "duplicate_result_shape_validation",
        "late_result_shape_validation",
        "late_input_validation",
        "extra_protected_torch_empty",
        "hidden_pre_parallel_declaration",
        "residual_structured_hoist",
        "policy_declaration_after_loop",
        "opaque_parallel_policy",
        "unknown_parallel_policy_macro",
        "parallel_schedule_subclass",
        "malformed_loop_flag",
        "malformed_conditional_flag",
        "malformed_top_level",
    ],
)
def test_heap_completion_owns_the_write_effect_and_function_state(
    monkeypatch, mutation
):
    """Completion recognizes the complete effect, not just access metadata."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = heap_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def corrupting(self, function):
        completed = original(self, function)
        write = next(
            node
            for node in relayout_llir_nodes(function.body)
            if type(node) is llir.ArrayAccess
            and type(getattr(node, "tensor_access", None)) is llir.TensorAccessMetadata
            and node.tensor_access.role is llir.TensorAccessRole.RESULT_WRITE
        )
        owner = next(
            node
            for node in relayout_llir_nodes(function.body)
            if type(node) is llir.Assign and node.var is write
        )
        if mutation == "wrong_assignment_op":
            owner.op = llir.AssignOp.ASSIGN
        elif mutation in ("unowned_physical_write", "unowned_top_level_write"):
            duplicate = copy.deepcopy(owner)
            assert type(duplicate.var) is llir.ArrayAccess
            object.__setattr__(duplicate.var, "tensor_access", None)
            if mutation == "unowned_physical_write":
                located = self._locate_statement(function.body, owner)
                assert located is not None
                body, position = located
                body.insert(position + 1, duplicate)
            else:
                function.body.append(duplicate)
        elif mutation == "opaque_physical_write":
            function.body.append(llir.RawStmt("C_values[0] = 123"))
        elif mutation in (
            "structured_tensor_alias",
            "structured_tensor_mutation",
            "opaque_call_name",
            "opaque_call_expression_name",
            "opaque_initializer_operator",
            "opaque_variable_name",
            "opaque_variable_type",
            "opaque_literal_text",
            "multiline_comment",
            "continued_comment",
            "initializer_operator_subclass",
            "unary_operator_subclass",
            "opaque_binary_operator",
            "binary_operator_subclass",
            "duplicate_tile_width_declaration",
            "duplicate_result_stack_declaration",
            "undeclared_variable",
            "use_before_declaration",
            "branch_local_variable_escape",
            "nested_function",
            "top_level_break",
            "top_level_continue",
            "missing_conditional_condition",
            "empty_conditional_then",
            "missing_conditional_branches",
            "mismatched_conditional_branches",
            "mutate_result_extent",
            "mutate_result_position",
            "mutate_result_shape",
            "move_result_shape",
            "escape_result_shape_address",
            "unowned_result_pointer_call",
            "guarded_call_protected_argument",
            "member_call_protected_argument",
            "member_call_expression_protected_argument",
            "forged_result_shape_validation",
            "duplicate_result_shape_validation",
            "hidden_pre_parallel_declaration",
            "residual_structured_hoist",
            "policy_declaration_after_loop",
            "opaque_parallel_policy",
            "unknown_parallel_policy_macro",
            "parallel_schedule_subclass",
            "malformed_loop_flag",
            "malformed_conditional_flag",
        ):
            final_assembly = next(
                index
                for index, stmt in enumerate(function.body)
                if type(stmt) is llir.Comment and stmt.value == "Assemble final result"
            )
            if mutation == "structured_tensor_alias":
                pointer_init = next(
                    stmt
                    for stmt in function.body
                    if type(stmt) is llir.VarInit
                    and type(stmt.var) is llir.Var
                    and stmt.var.name == "C_values"
                )
                alias_init = copy.deepcopy(pointer_init)
                alias_init.var.name = "C_values_alias"
                alias_write = llir.Assign(
                    llir.ArrayAccess(
                        llir.Var("C_values_alias", alias_init.var.type),
                        llir.Literal(0, llir.DataType.INT64),
                    ),
                    llir.Literal(123.0, llir.DataType.FLOAT32),
                )
                function.body[final_assembly:final_assembly] = [
                    alias_init,
                    alias_write,
                ]
            elif mutation == "structured_tensor_mutation":
                function.body.insert(
                    final_assembly,
                    llir.MemberCallStmt(
                        base=llir.Var(
                            "C_values_torch",
                            llir.DataType.TORCH_TENSOR,
                        ),
                        member="zero_",
                    ),
                )
            elif mutation == "opaque_call_name":
                function.body.insert(
                    final_assembly,
                    llir.FunctionCallStmt("C_values_torch.zero_"),
                )
            elif mutation == "opaque_call_expression_name":
                function.body.insert(
                    final_assembly,
                    llir.VarInit(
                        llir.Var("decoy", llir.DataType.NO_TYPE),
                        llir.FunctionCall("C_values_torch.zero_"),
                    ),
                )
            elif mutation == "opaque_initializer_operator":
                function.body.insert(
                    final_assembly,
                    llir.VarInit(
                        llir.Var("decoy", llir.DataType.INT),
                        llir.Literal(0),
                        op="= 0; C_values_torch.zero_(); int decoy2 =",
                    ),
                )
            elif mutation == "opaque_variable_name":
                function.body[final_assembly:final_assembly] = [
                    llir.VarInit(
                        llir.Var("decoy", llir.DataType.INT),
                        llir.Literal(0),
                    ),
                    llir.Assign(
                        llir.Var("decoy", llir.DataType.INT),
                        llir.Var(
                            "0; C_values_torch.zero_(); decoy",
                            llir.DataType.INT,
                        ),
                    ),
                ]
            elif mutation == "opaque_variable_type":
                forged_type = type(
                    "ForgedDataType",
                    (),
                    {"value": ("int decoy; C_values_torch.zero_(); int")},
                )()
                function.body.insert(
                    final_assembly,
                    llir.VarDecl(llir.Var("decoy2", forged_type)),
                )
            elif mutation == "opaque_literal_text":
                function.body.insert(
                    final_assembly,
                    llir.VarInit(
                        llir.Var("decoy", llir.DataType.INT),
                        llir.Literal(
                            "0; C_values_torch.zero_(); int decoy2 = 0",
                            llir.DataType.INT,
                        ),
                    ),
                )
            elif mutation == "multiline_comment":
                function.body.insert(
                    final_assembly,
                    llir.Comment("decoy\nC_values_torch.zero_();"),
                )
            elif mutation == "continued_comment":
                located = self._locate_statement(function.body, owner)
                assert located is not None
                body, position = located
                body.insert(position, llir.Comment("suppress result write \\"))
            elif mutation == "initializer_operator_subclass":

                class EffectfulOperator(str):
                    def __format__(self, format_spec):
                        return "= 0; C_values_torch.zero_(); int decoy2 ="

                function.body.insert(
                    final_assembly,
                    llir.VarInit(
                        llir.Var("decoy", llir.DataType.INT),
                        llir.Literal(0),
                        op=EffectfulOperator("="),
                    ),
                )
            elif mutation == "unary_operator_subclass":

                class EffectfulUnaryOperator(str):
                    def __format__(self, format_spec):
                        return "- 0; C_values_torch.zero_(); int decoy2 ="

                function.body.insert(
                    final_assembly,
                    llir.VarInit(
                        llir.Var("decoy", llir.DataType.INT),
                        llir.UnaryOp(
                            EffectfulUnaryOperator("-"),
                            llir.Literal(0),
                        ),
                    ),
                )
            elif mutation == "opaque_binary_operator":
                function.body.insert(
                    final_assembly,
                    llir.VarInit(
                        llir.Var("decoy", llir.DataType.INT),
                        llir.BinOp(
                            "+ 0; C_values_torch.zero_(); int decoy2 = 0 +",
                            llir.Literal(0, llir.DataType.INT),
                            llir.Literal(0, llir.DataType.INT),
                        ),
                    ),
                )
            elif mutation == "binary_operator_subclass":

                class EffectfulBinaryOperator(str):
                    def __format__(self, format_spec):
                        return "+ 0; C_values_torch.zero_(); int decoy2 = 0 +"

                spoofed = llir.BinOp(
                    "+",
                    llir.Literal(0, llir.DataType.INT),
                    llir.Literal(0, llir.DataType.INT),
                )
                object.__setattr__(spoofed, "op", EffectfulBinaryOperator("+"))
                function.body.insert(
                    final_assembly,
                    llir.VarInit(llir.Var("decoy", llir.DataType.INT), spoofed),
                )
            elif mutation == "duplicate_tile_width_declaration":
                tile_width = next(
                    stmt
                    for stmt in function.body
                    if type(stmt) is llir.VarInit
                    and type(stmt.var) is llir.Var
                    and stmt.var.name.startswith("kTile_")
                )
                pack_origin = next(
                    stmt for stmt in function.body if type(stmt) is llir.ForLoop
                )
                pack_origin.body.insert(
                    0,
                    llir.VarInit(
                        llir.Var(tile_width.var.name, tile_width.var.type),
                        llir.Literal(1),
                    ),
                )
            elif mutation == "duplicate_result_stack_declaration":
                pack_origin = next(
                    stmt for stmt in function.body if type(stmt) is llir.ForLoop
                )
                pack_origin.body.insert(
                    0,
                    llir.FixedStackArrayDecl(
                        name="C_values",
                        element_type=llir.DataType.FLOAT32,
                        extent=llir.Literal(1),
                        initializer=llir.Array((), llir.DataType.FLOAT32),
                    ),
                )
            elif mutation == "undeclared_variable":
                function.body.insert(
                    final_assembly,
                    llir.VarInit(
                        llir.Var("review_sink", llir.DataType.INT),
                        llir.Var("review_ghost", llir.DataType.INT),
                    ),
                )
            elif mutation == "use_before_declaration":
                function.body[final_assembly:final_assembly] = [
                    llir.VarInit(
                        llir.Var("review_sink", llir.DataType.INT),
                        llir.Var("review_later", llir.DataType.INT),
                    ),
                    llir.VarInit(
                        llir.Var("review_later", llir.DataType.INT),
                        llir.Literal(1),
                    ),
                ]
            elif mutation == "branch_local_variable_escape":
                function.body[final_assembly:final_assembly] = [
                    llir.IfThenElse(
                        cond=llir.Literal(True),
                        then_body=[
                            llir.VarInit(
                                llir.Var("review_branch", llir.DataType.INT),
                                llir.Literal(1),
                            )
                        ],
                    ),
                    llir.VarInit(
                        llir.Var("review_sink", llir.DataType.INT),
                        llir.Var("review_branch", llir.DataType.INT),
                    ),
                ]
            elif mutation == "nested_function":
                function.body.insert(
                    final_assembly,
                    llir.Function(
                        llir.DataType.VOID,
                        "review_nested",
                        (),
                        [],
                    ),
                )
            elif mutation in _HEAP_REVIEW_CONTROL_AND_EFFECT_MUTATIONS:
                _apply_heap_review_control_or_effect_mutation(
                    self,
                    function,
                    owner,
                    final_assembly,
                    mutation,
                )
            elif mutation == "hidden_pre_parallel_declaration":
                loop = next(
                    stmt
                    for stmt in relayout_llir_nodes(function.body)
                    if type(stmt) is llir.ForLoop
                    and not stmt.omp_parallel_for
                    and not getattr(stmt, "_use_atomic_scheduling", False)
                )
                loop.pre_parallel_body = [
                    llir.VarInit(
                        llir.Var("review_hidden", llir.DataType.INT),
                        llir.Literal(1),
                    )
                ]
            elif mutation == "residual_structured_hoist":
                loop = next(
                    stmt
                    for stmt in relayout_llir_nodes(function.body)
                    if type(stmt) is llir.ForLoop
                )
                loop._hoisted_ptr_decls = [
                    llir.VarInit(
                        llir.Var("review_hoist", llir.DataType.INT),
                        llir.Literal(1),
                    )
                ]
            elif mutation == "policy_declaration_after_loop":
                loop = next(
                    stmt
                    for stmt in relayout_llir_nodes(function.body)
                    if type(stmt) is llir.ForLoop
                )
                loop.omp_parallel_for = True
                loop.omp_num_threads = "scorch_nthreads(review_later_policy, A0_size)"
                function.body.insert(
                    final_assembly,
                    llir.VarInit(
                        llir.Var("review_later_policy", llir.DataType.INT),
                        llir.Literal(1),
                    ),
                )
            elif mutation == "malformed_conditional_flag":
                conditional = next(
                    stmt
                    for stmt in relayout_llir_nodes(function.body)
                    if type(stmt) is llir.IfThenElse
                )
                conditional.make_last_case_else = object()
            else:
                loop = next(
                    stmt
                    for stmt in relayout_llir_nodes(function.body)
                    if type(stmt) is llir.ForLoop
                )
                loop.omp_parallel_for = True
                if mutation == "opaque_parallel_policy":
                    loop.omp_num_threads = (
                        "scorch_nthreads((C_values_torch.zero_(), 1), 1)"
                    )
                elif mutation == "unknown_parallel_policy_macro":
                    loop.omp_num_threads = "scorch_nthreads(UNKNOWN_EFFECT, A0_size)"
                elif mutation == "parallel_schedule_subclass":

                    class EffectfulSchedule(str):
                        def __format__(self, format_spec):
                            return "static)\nC_values_torch.zero_();\n#pragma omp for"

                    loop.omp_schedule = EffectfulSchedule("static")
                else:
                    loop.simd = 1
        else:
            malformed = next(
                stmt for stmt in function.body if type(stmt) is llir.VarInit
            )
            object.__delattr__(malformed, "var")
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", corrupting)
    expect_target_code(
        "result_tile_completion_lost",
        fixture.program,
        heap_shapes(fixture),
        (4, 6),
    )


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "moved", "wrong_arguments"]
)
def test_heap_completion_requires_exactly_one_generated_zero(monkeypatch, mutation):
    """The copy-out coverage proof requires exactly one dense-result zero."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = heap_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def mutating(self, function):
        completed = original(self, function)
        zero_calls = [
            (index, stmt)
            for index, stmt in enumerate(function.body)
            if isinstance(stmt, llir.FunctionCallStmt)
            and stmt.name == "scorch_zero_dense"
        ]
        assert len(zero_calls) == 1
        index, stmt = zero_calls[0]
        if mutation == "missing":
            del function.body[index]
        elif mutation == "duplicate":
            function.body.insert(index, copy.deepcopy(stmt))
        elif mutation == "moved":
            del function.body[index]
            outer = next(
                candidate
                for candidate in function.body
                if type(candidate) is llir.ForLoop
            )
            outer.body.append(stmt)
        else:
            object.__setattr__(
                stmt,
                "args",
                (stmt.args[0], llir.Literal(1, llir.DataType.INT64)),
            )
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", mutating)
    expect_target_code(
        "result_tile_completion_lost",
        fixture.program,
        heap_shapes(fixture),
        (4, 6),
    )


def test_heap_completion_rejects_a_corrupted_chain(monkeypatch):
    """A post-pass header mutation of the pack origin fails re-identification."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = heap_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def corrupting(self, function):
        completed = original(self, function)
        outer = next(stmt for stmt in function.body if isinstance(stmt, llir.ForLoop))
        outer.omp_schedule = "static"
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", corrupting)
    expect_target_code(
        "result_tile_completion_lost",
        fixture.program,
        heap_shapes(fixture),
        (4, 6),
    )


def multi_prefix_heap_program(strip=3, dtype=None):
    from scorch.compiler.loopir.nodes import ScalarType as _ScalarType

    from tests.test_scorch.test_loopir_verifier import build_heap_ttm

    return build_heap_ttm(
        strip=strip,
        dtype=dtype if dtype is not None else _ScalarType.FLOAT32,
    )


def multi_prefix_heap_shapes(fixture):
    return {fixture.core: (3, 4, 5), fixture.factor: (5, 6)}


def test_multi_prefix_heap_target_emits_the_legacy_compact_source():
    """Direct structural activation on a bare verified rank-3 program."""

    fixture = multi_prefix_heap_program()
    lowered = lower_loopir_to_llir(
        fixture.program,
        input_shapes=multi_prefix_heap_shapes(fixture),
        result_shape=(3, 4, 6),
    )
    from scorch.compiler.codegen import LLIRLowerer

    source = LLIRLowerer().lower_llir(lowered)
    # The compact extent is the product of every dense prefix level.
    assert (
        "std::vector<float> tiled_Projected_storage("
        "(size_t)(Projected0_size * Projected1_size) * (size_t)kTile_d);" in source
    )
    assert (
        "float* __restrict__ tiled_Projected = tiled_Projected_storage.data();"
        in source
    )
    assert "// Initialize compact result tile for Projected" in source
    assert "// Copy compact result tile to Projected" in source
    # The compact row is the linearized position of the last prefix level.
    assert (
        "tiled_Projected[pProjected1 * kTile_d + d_in] += "
        "Core_val[pCore2] * Factor_val[pFactor1];" in source
    )
    assert (
        "tiled_Projected[Projected_tile_init * kTile_d + d_tile_init] = 0.0f;" in source
    )
    assert (
        "Projected_values[Projected_tile_copy * Projected2_size + d_copy_logical] = "
        "tiled_Projected[Projected_tile_copy * kTile_d + d_tile_copy];" in source
    )
    assert "scorch_zero_dense(Projected_values" not in source
    # The parallel policy lands on the outermost dense prefix loop.
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(-1, Core0_size)) "
        "schedule(dynamic, scorch_chunk(Core0_size, -1))" in source
    )


def test_multi_prefix_heap_target_rejects_a_permuted_prefix_chain():
    """The prefix loops must appear in the result's physical storage order."""

    from tests.test_scorch.test_loopir_verifier import forge

    fixture = multi_prefix_heap_program()
    result_decl = fixture.program.tensors[2]
    # Swap the result's two dense prefix dimensions without moving the
    # chain: the loop at prefix position 0 no longer stores mode 0.
    forge(
        result_decl,
        dimensions=(fixture.dim_b, fixture.dim_a, fixture.dim_d),
    )
    forge(
        fixture.leaf,
        indices=(
            fixture.builder.index_value(fixture.row),
            fixture.builder.index_value(fixture.batch),
            fixture.builder.index_value(fixture.free),
        ),
    )
    expect_target_code(
        "unsupported_program_shape",
        fixture.program,
        multi_prefix_heap_shapes(fixture),
        (4, 3, 6),
    )


def test_heap_target_rejects_a_misplaced_region():
    """A repeating result-tile lifetime is rejected by the shared verifier."""

    from tests.test_scorch.test_loopir_verifier import forge

    fixture = heap_program()
    # Move the region below the row loop: region(body=[sparse chain]) sits
    # inside the row loop instead of wrapping the pack origin's body.
    row_loop = fixture.region.body.statements[0]
    inner_region = fixture.builder.result_tile_region(fixture.decl, row_loop.body)
    forge(row_loop, body=fixture.builder.block((inner_region,)))
    forge(fixture.pack, body=fixture.builder.block((row_loop,)))
    with pytest.raises(LoopIRVerificationError) as captured:
        lower_loopir_to_llir(
            fixture.program,
            input_shapes=heap_shapes(fixture),
            result_shape=(4, 6),
        )
    assert captured.value.defect.code == "result_tile_scope_mismatch"


@pytest.mark.parametrize("hidden_name", ["atomic_chunk", "atomic_bound"])
def test_heap_binding_rejects_atomic_names_before_declaration(hidden_name):
    """Synthetic atomic names are invisible until codegen declares them."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    chunk = llir.Var("chunk", llir.DataType.INT)
    bound = llir.Var("bound", llir.DataType.INT)
    loop = llir.ForLoop(
        init=llir.VarInit(llir.Var("i", llir.DataType.INT), llir.Literal(0)),
        cond=llir.BinOp("<", llir.Var("i", llir.DataType.INT), bound),
        update=llir.Increment(llir.Var("i", llir.DataType.INT)),
        body=[llir.Continue()],
    )
    loop._use_atomic_scheduling = True
    loop._atomic_counter_var = "counter"
    if hidden_name == "atomic_chunk":
        loop._atomic_chunk_var = "review_ghost_chunk"
        loop._loop_bound = "_start"
        expected = "ForLoop._atomic_chunk_var references 'review_ghost_chunk'"
    else:
        loop._atomic_chunk_var = "chunk"
        loop._loop_bound = "review_ghost_bound"
        expected = "ForLoop._loop_bound references 'review_ghost_bound'"
    loop.omp_num_threads = "scorch_nthreads(counter, bound)"
    function = llir.Function(
        llir.DataType.VOID,
        "evaluate",
        (chunk, bound),
        [loop],
    )
    with pytest.raises(ValueError) as captured:
        lower_llir_module._validate_result_tile_rendered_text(
            function,
            protected_names=set(),
        )
    assert expected in str(captured.value)
    assert "before a visible declaration" in str(captured.value)


@pytest.mark.parametrize(
    "mutation",
    [
        "prefix_position_increment",
        "inner_position_assign",
        "comment_splice",
        "undeclared_variable",
    ],
)
def test_multi_prefix_heap_completion_owns_protected_state(monkeypatch, mutation):
    """The rank-3 kernel's own position and text boundaries are owned."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = multi_prefix_heap_program()
    original = lower_llir_module._TargetLowering.complete_panel

    def corrupting(self, function):
        completed = original(self, function)
        write = next(
            node
            for node in relayout_llir_nodes(function.body)
            if type(node) is llir.ArrayAccess
            and type(getattr(node, "tensor_access", None)) is llir.TensorAccessMetadata
            and node.tensor_access.role is llir.TensorAccessRole.RESULT_WRITE
        )
        owner = next(
            node
            for node in relayout_llir_nodes(function.body)
            if type(node) is llir.Assign and node.var is write
        )
        located = self._locate_statement(function.body, owner)
        assert located is not None
        body, position = located
        if mutation == "prefix_position_increment":
            body.insert(
                position,
                llir.Increment(llir.Var("pProjected1", llir.DataType.INT)),
            )
        elif mutation == "inner_position_assign":
            body.insert(
                position,
                llir.Assign(
                    llir.Var("pProjected2", llir.DataType.INT),
                    llir.Literal(0, llir.DataType.INT),
                ),
            )
        elif mutation == "comment_splice":
            body.insert(
                position,
                llir.Comment("decoy\nProjected_values_torch.zero_();"),
            )
        else:
            assert mutation == "undeclared_variable"
            body.insert(
                position,
                llir.VarInit(
                    llir.Var("review_sink", llir.DataType.INT),
                    llir.Var("review_ghost", llir.DataType.INT),
                ),
            )
        return completed

    monkeypatch.setattr(lower_llir_module._TargetLowering, "complete_panel", corrupting)
    expect_target_code(
        "result_tile_completion_lost",
        fixture.program,
        multi_prefix_heap_shapes(fixture),
        (3, 4, 6),
    )


# -- Abstract parallel selection on bare programs ------------------------------


def bare_matmul_with_selection():
    """The verifier matmul fixture with an explicit selection of ``j``."""

    from tests.test_scorch.test_loopir_verifier import (
        attach_selection,
        build_matmul as build_bare_matmul,
    )

    fixture = build_bare_matmul()
    loop_i = fixture.program.body.statements[0]
    loop_k = loop_i.body.statements[0]
    loop_j = loop_k.body.statements[0]
    attach_selection(fixture, loop_j.index, rows=loop_j.dimension)
    return fixture


def bare_matmul_shapes(fixture):
    a_symbol, b_symbol = fixture.program.inputs
    return {a_symbol: (4, 5), b_symbol: (5, 6)}


def test_bare_parallel_selection_marks_the_selected_loop():
    """Direct structural activation: the selection marks ``j``, not the row."""

    from scorch.compiler.codegen import LLIRLowerer

    fixture = bare_matmul_with_selection()
    lowered = lower_loopir_to_llir(
        fixture.program,
        input_shapes=bare_matmul_shapes(fixture),
        result_shape=(4, 6),
    )
    source = LLIRLowerer().lower_llir(lowered)
    assert (
        "#pragma omp parallel for num_threads(scorch_nthreads(-1, B1_size)) "
        "schedule(dynamic, scorch_chunk(B1_size, -1))"
    ) in source
    # The auto gate stayed suppressed: the row loop carries no policy.
    assert "scorch_nthreads(-1, A0_size)" not in source


def test_bare_heap_selection_adopts_the_inner_prefix_anchor():
    """A bare rank-3 heap program realizes the lifted inner-prefix anchor."""

    from scorch.compiler.codegen import LLIRLowerer
    from scorch.compiler.loopir.nodes import ParallelDiscipline

    from tests.test_scorch.test_loopir_verifier import (
        attach_selection,
        build_heap_ttm,
    )

    fixture = build_heap_ttm()
    attach_selection(
        fixture,
        fixture.row,
        discipline=ParallelDiscipline.COMPACT_PARTITION,
        rows=fixture.dim_b,
        nnz=fixture.builder.sparse_work_source(fixture.core, 2),
    )
    lowered = lower_loopir_to_llir(
        fixture.program,
        input_shapes={fixture.core: (3, 4, 5), fixture.factor: (5, 6)},
        result_shape=(3, 4, 6),
    )
    source = LLIRLowerer().lower_llir(lowered)
    assert "num_threads(scorch_nthreads(Core2_pos[Core1_size], Core1_size))" in source
    assert "num_threads(scorch_nthreads(-1, Core0_size))" not in source


def test_heap_completion_revalidates_the_owned_work_fact(monkeypatch):
    from scorch.compiler.loopir import lower_llir as lower_llir_module
    from scorch.compiler.loopir.nodes import ParallelDiscipline

    from tests.test_scorch.test_loopir_verifier import (
        attach_selection,
        build_heap_ttm,
        forge,
    )

    fixture = build_heap_ttm()
    attach_selection(
        fixture,
        fixture.row,
        discipline=ParallelDiscipline.COMPACT_PARTITION,
        rows=fixture.dim_b,
        nnz=fixture.builder.sparse_work_source(fixture.core, 2),
    )
    original = lower_llir_module._TargetLowering.complete_result_tile

    def mutate_work(self, function):
        assert self.parallel is not None
        forge(self.parallel.work, nnz=None)
        return original(self, function)

    monkeypatch.setattr(
        lower_llir_module._TargetLowering,
        "complete_result_tile",
        mutate_work,
    )
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            fixture.program,
            input_shapes={fixture.core: (3, 4, 5), fixture.factor: (5, 6)},
            result_shape=(3, 4, 6),
        )
    assert error.value.defect.code == "result_tile_completion_lost"


@pytest.mark.parametrize("mutation", ["work_fact", "retained_row_policy"])
def test_heap_panel_handoff_revalidates_parallel_state(monkeypatch, mutation):
    from scorch.compiler.loopir import lower_llir as lower_llir_module
    from scorch.compiler.loopir.schedule_passes import apply_schedule_plan

    from tests.test_scorch.test_loopir_schedule_passes import (
        build_ttm_abcd,
        lower,
        multi_prefix_heap_panel_plan,
    )
    from tests.test_scorch.test_loopir_verifier import forge

    cin, (a, b, c, d) = build_ttm_abcd()
    lowering = lower(cin)
    plan = multi_prefix_heap_panel_plan(lowering, a, b, c, d, anchor=b)
    artifact = apply_schedule_plan(lowering.program, plan)
    original = lower_llir_module._TargetLowering.complete_result_tile

    def corrupt_handoff(self, function):
        assert self.parallel is not None
        assert self._panel_completion is not None
        if mutation == "work_fact":
            forge(self.parallel.work, nnz=None)
        else:
            row_loop = self._panel_completion[2]
            row_loop.omp_num_threads = "scorch_nthreads(-1, Core1_size)"
            row_loop.omp_chunk_expr = "scorch_chunk(Core1_size, -1)"
        return original(self, function)

    monkeypatch.setattr(
        lower_llir_module._TargetLowering,
        "complete_result_tile",
        corrupt_handoff,
    )
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            artifact.program,
            input_shapes={
                lowering.input_symbols[0]: (3, 4, 5),
                lowering.input_symbols[1]: (5, 6),
            },
            result_shape=(3, 4, 6),
        )
    assert error.value.defect.code == "result_tile_completion_lost"


def test_relayout_handoff_revalidates_the_panel_row_policy(monkeypatch):
    from scorch.compiler.loopir import lower_llir as lower_llir_module

    from tests.test_scorch.test_loopir_schedule_passes import scheduled_relayout

    lowering, _indices, _plan, artifact = scheduled_relayout()
    original = lower_llir_module._TargetLowering.complete_relayout

    def corrupt_handoff(self, function):
        assert self._panel_completion is not None
        row_loop = self._panel_completion[2]
        row_loop.omp_num_threads = "scorch_nthreads(-1, A0_size)"
        row_loop.omp_chunk_expr = "scorch_chunk(A0_size, -1)"
        return original(self, function)

    monkeypatch.setattr(
        lower_llir_module._TargetLowering,
        "complete_relayout",
        corrupt_handoff,
    )
    with pytest.raises(LoopIRTargetError) as error:
        lower_loopir_to_llir(
            artifact.program,
            input_shapes={
                lowering.input_symbols[0]: (4, 5),
                lowering.input_symbols[1]: (5, 6),
            },
            result_shape=(4, 6),
        )
    assert error.value.defect.code == "relayout_completion_lost"


@pytest.mark.parametrize(
    "mutation",
    [
        "premarked_selection",
        "missing_selection_snapshot",
        "mutated_selection_header",
        "forged_work_source",
        "undeclared_work_source",
        "relocated_selection",
        "premarked_nonselected",
        "atomic_nonselected",
        "mutated_ancestor_header",
    ],
)
def test_parallel_completion_owns_the_selected_loop(monkeypatch, mutation):
    """Post-assembly loss of the selection is stage-owned, never guessed."""

    from scorch.compiler.loopir import lower_llir as lower_llir_module

    fixture = bare_matmul_with_selection()
    original = lower_llir_module._TargetLowering.complete_parallel

    def corrupting(self, function):
        if mutation == "missing_selection_snapshot":
            self._emitted_loop_headers.clear()
        elif mutation == "forged_work_source":
            from tests.test_scorch.test_loopir_verifier import forge

            forge(
                self.parallel.work,
                nnz=fixture.builder.sparse_work_source(fixture.program.inputs[0], 1),
            )
        elif mutation == "undeclared_work_source":
            from tests.test_scorch.test_loopir_verifier import forge

            forge(
                self.parallel.work,
                nnz=fixture.builder.sparse_work_source(
                    fixture.builder.new_symbol_id(),
                    1,
                ),
            )
        else:
            target = None
            for node in relayout_llir_nodes(function.body):
                if (
                    type(node) is llir.ForLoop
                    and node.init is not None
                    and type(node.init.var) is llir.Var
                    and node.init.var.name == "j"
                ):
                    target = node
            assert target is not None
            if mutation == "premarked_selection":
                target.omp_parallel_for = True
            elif mutation == "relocated_selection":
                located = self._locate_statement(function.body, target)
                assert located is not None and located[0] is not function.body
                owner, position = located
                del owner[position]
                final_assembly = next(
                    index
                    for index, statement in enumerate(function.body)
                    if type(statement) is llir.Comment
                    and statement.value == "Assemble final result"
                )
                function.body.insert(final_assembly, target)
            elif mutation == "premarked_nonselected":
                reduction = next(
                    node
                    for node in relayout_llir_nodes(function.body)
                    if type(node) is llir.ForLoop
                    and type(node.init) is llir.VarInit
                    and type(node.init.var) is llir.Var
                    and node.init.var.name == "k"
                )
                reduction.omp_parallel_for = True
            elif mutation == "atomic_nonselected":
                reduction = next(
                    node
                    for node in relayout_llir_nodes(function.body)
                    if type(node) is llir.ForLoop
                    and type(node.init) is llir.VarInit
                    and type(node.init.var) is llir.Var
                    and node.init.var.name == "k"
                )
                reduction._use_atomic_scheduling = True
            elif mutation == "mutated_ancestor_header":
                ancestor = next(
                    node
                    for node in relayout_llir_nodes(function.body)
                    if type(node) is llir.ForLoop
                    and type(node.init) is llir.VarInit
                    and type(node.init.var) is llir.Var
                    and node.init.var.name == "i"
                )
                ancestor.init.var.name = "i_review"
            else:
                assert mutation == "mutated_selection_header"
                target.init.var.name = "j_review"
        return original(self, function)

    monkeypatch.setattr(
        lower_llir_module._TargetLowering, "complete_parallel", corrupting
    )
    with pytest.raises(LoopIRTargetError) as captured:
        lower_loopir_to_llir(
            fixture.program,
            input_shapes=bare_matmul_shapes(fixture),
            result_shape=(4, 6),
        )
    assert captured.value.defect.code == "parallel_completion_lost"
