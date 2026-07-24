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
from scorch.compiler.loop_plan import MAX_AFFINE_TILE_WIDTH
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
    fixture = build_matmul()
    decl = fixture.program.tensors[0]
    forge(decl.levels[0], mode=1)
    forge(decl.levels[1], mode=0)
    expect_target_code(
        "unsupported_mode_order",
        fixture.program,
        {fixture.a: (3, 5), fixture.b: (5, 4)},
        (3, 4),
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


def test_stack_region_lowers_through_the_target():
    fixture = build_stack_matmul(width=4)
    shapes, result_shape = stack_matmul_shapes(fixture)
    function = lower_loopir_to_llir(
        fixture.program, input_shapes=shapes, result_shape=result_shape
    )
    assert function is not None


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
