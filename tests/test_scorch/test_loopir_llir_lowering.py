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


def test_invalid_display_name():
    program = build_matvec()
    forge(program.tensors[0], name="A values")
    expect_target_code("invalid_display_name", program, matvec_shapes(program), (3,))


def test_duplicate_display_name():
    program = build_matvec()
    forge(program.tensors[1], name="A")
    expect_target_code("duplicate_display_name", program, matvec_shapes(program), (3,))


def test_dimension_and_tensor_names_share_one_namespace():
    program = build_matvec()
    forge(program.tensors[0], name="i")
    expect_target_code("duplicate_display_name", program, matvec_shapes(program), (3,))


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
