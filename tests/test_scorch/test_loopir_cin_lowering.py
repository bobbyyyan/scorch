"""Normalized-CIN-to-LoopIR lowering: structure, provenance, and boundaries.

The lowering must map the dense elementwise and reduction/matmul families
onto LoopIR with stable identities preserved, produce a verified LoopPlan,
never mutate its input, and fail closed with stable codes on everything
outside the families.
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
    UnaryOp,
    Where,
    Workspace,
)
from scorch.compiler.cin_analysis import canonical_cin_dump, normalize_cin
from scorch.compiler.diagnostics import VerificationError
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.lower_cin import (
    LoopIRLoweringError,
    lower_normalized_cin_to_loopir,
)
from scorch.compiler.loopir.nodes import (
    BinaryOp,
    DenseFor,
    LevelKind,
    ReduceOp,
    ScalarType,
    Store,
    StoreReduce,
)
from scorch.compiler.loopir.printer import canonical_program_dump
from scorch.compiler.loopir.verifier import verify_program


def build_elementwise_add():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]))
    return ForAll(i, ForAll(j, assign)), (i, j), (a, b, c)


def build_matmul_ikj():
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=Operation.ADD
    )
    return ForAll(i, ForAll(k, ForAll(j, assign))), (i, k, j), (a, b, c)


def expect_code(code, cin):
    with pytest.raises(LoopIRLoweringError) as error:
        lower_normalized_cin_to_loopir(normalize_cin(cin))
    assert error.value.defect.code == code, error.value.defect
    return error.value.defect


def test_elementwise_structure_and_provenance():
    cin, (i, j), (a, b, c) = build_elementwise_add()
    result = lower_normalized_cin_to_loopir(normalize_cin(cin))
    program = result.program
    verify_program(program)

    assert result.loop_index_ids == (i.index_id, j.index_id)
    assert result.loop_plan.loop_order == (i.index_id, j.index_id)
    assert result.input_symbols == (a.symbol_id, b.symbol_id)
    assert result.rhs_access_symbols == (a.symbol_id, b.symbol_id)
    assert result.result_symbol == c.symbol_id

    assert tuple(decl.symbol for decl in program.tensors) == (
        a.symbol_id,
        b.symbol_id,
        c.symbol_id,
    )
    assert all(decl.dtype is ScalarType.FLOAT32 for decl in program.tensors)
    assert all(
        level.kind is LevelKind.DENSE
        for decl in program.tensors
        for level in decl.levels
    )

    outer = program.body.statements[0]
    assert type(outer) is DenseFor and outer.index == i.index_id
    inner = outer.body.statements[0]
    assert type(inner) is DenseFor and inner.index == j.index_id
    leaf = inner.body.statements[0]
    assert type(leaf) is Store
    assert leaf.tensor == c.symbol_id
    assert leaf.value.op is BinaryOp.ADD


def test_matmul_structure():
    cin, (i, k, j), (a, b, c) = build_matmul_ikj()
    result = lower_normalized_cin_to_loopir(normalize_cin(cin))
    leaf = (
        result.program.body.statements[0]
        .body.statements[0]
        .body.statements[0]
        .body.statements[0]
    )
    assert type(leaf) is StoreReduce
    assert leaf.op is ReduceOp.ADD
    assert leaf.value.op is BinaryOp.MUL
    assert result.loop_plan.loop_order == (i.index_id, k.index_id, j.index_id)


def test_float64_and_sub_lower():
    i = IndexVar("i")
    a = TensorVar("a", fmt="d", dtype=torch.float64)
    b = TensorVar("b", fmt="d", dtype=torch.float64)
    c = TensorVar("c", fmt="d", dtype=torch.float64)
    assign = TensorAssign(c[i], CINBinaryOp(Operation.SUB, a[i], b[i]))
    result = lower_normalized_cin_to_loopir(normalize_cin(ForAll(i, assign)))
    assert all(decl.dtype is ScalarType.FLOAT64 for decl in result.program.tensors)
    leaf = result.program.body.statements[0].body.statements[0]
    assert leaf.value.op is BinaryOp.SUB


def test_lowering_matches_hand_built_equivalent_canonically():
    cin, _, _ = build_elementwise_add()
    lowered_dump = canonical_program_dump(
        lower_normalized_cin_to_loopir(normalize_cin(cin)).program
    )

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    decls = tuple(
        builder.tensor(
            symbol,
            name,
            ScalarType.FLOAT32,
            (dim_i.dimension, dim_j.dimension),
            builder.dense_levels(2),
        )
        for symbol, name in ((a, "A"), (b, "B"), (c, "C"))
    )
    index_i, index_j = builder.new_index_id(), builder.new_index_id()
    store = builder.store(
        c,
        (builder.index_value(index_i), builder.index_value(index_j)),
        builder.binary(
            BinaryOp.ADD,
            builder.load(
                a, (builder.index_value(index_i), builder.index_value(index_j))
            ),
            builder.load(
                b, (builder.index_value(index_i), builder.index_value(index_j))
            ),
        ),
    )
    inner = builder.dense_for(index_j, dim_j.dimension, builder.block((store,)))
    outer = builder.dense_for(index_i, dim_i.dimension, builder.block((inner,)))
    hand_built = builder.program(
        (dim_i, dim_j), decls, (a, b), (c,), builder.block((outer,))
    )
    assert canonical_program_dump(hand_built) == lowered_dump


def test_lowering_is_deterministic_and_does_not_mutate_input():
    cin, _, _ = build_elementwise_add()
    normalized = normalize_cin(cin)
    before = canonical_cin_dump(normalized)
    first = lower_normalized_cin_to_loopir(normalized)
    second = lower_normalized_cin_to_loopir(normalized)
    assert canonical_cin_dump(normalized) == before
    assert canonical_program_dump(first.program) == canonical_program_dump(
        second.program
    )


def test_where_fails_closed():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    x = TensorVar("x", fmt="d")
    y = TensorVar("y", fmt="d")
    workspace = Workspace(name="wksp", dim=0)
    producer = ForAll(
        j,
        TensorAssign(
            workspace.get_default_access(),
            CINBinaryOp(Operation.MUL, a[i, j], x[j]),
            op=Operation.ADD,
        ),
    )
    consumer = TensorAssign(y[i], workspace.get_default_access())
    cin = ForAll(i, Where(producer=producer, consumer=consumer))
    expect_code("unsupported_statement", cin)


def test_missing_loop_fails_closed():
    i = IndexVar("i")
    a = TensorVar("a", fmt="d")
    c = TensorVar("c", fmt="d")
    expect_code("unsupported_statement", TensorAssign(c[i], a[i]))


def test_explicit_parallel_mark_fails_closed():
    cin, (i, j), _ = build_elementwise_add()
    cin.parallel = True
    expect_code("unsupported_explicit_parallel", cin)


def test_unsupported_update_op():
    i = IndexVar("i")
    a = TensorVar("a", fmt="d")
    c = TensorVar("c", fmt="d")
    assign = TensorAssign(c[i], a[i], op=Operation.MUL)
    expect_code("unsupported_update_op", ForAll(i, assign))


def test_unsupported_operation_div():
    i = IndexVar("i")
    a = TensorVar("a", fmt="d")
    b = TensorVar("b", fmt="d")
    c = TensorVar("c", fmt="d")
    assign = TensorAssign(c[i], CINBinaryOp(Operation.DIV, a[i], b[i]))
    expect_code("unsupported_operation", ForAll(i, assign))


def test_unsupported_expression_unary():
    i = IndexVar("i")
    a = TensorVar("a", fmt="d")
    c = TensorVar("c", fmt="d")
    assign = TensorAssign(c[i], UnaryOp(Operation.SUB, a[i]))
    expect_code("unsupported_expression", ForAll(i, assign))


def test_unsupported_format():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="ds")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    expect_code("unsupported_format", ForAll(i, ForAll(j, assign)))


def test_unsupported_mode_order():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd", mode_order=[1, 0])
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    expect_code("unsupported_mode_order", ForAll(i, ForAll(j, assign)))


def test_unsupported_dtype():
    i = IndexVar("i")
    a = TensorVar("a", fmt="d", dtype=torch.int32)
    c = TensorVar("c", fmt="d", dtype=torch.int32)
    assign = TensorAssign(c[i], a[i])
    expect_code("unsupported_dtype", ForAll(i, assign))


def test_mixed_dtype():
    i = IndexVar("i")
    a = TensorVar("a", fmt="d", dtype=torch.float64)
    b = TensorVar("b", fmt="d", dtype=torch.float32)
    c = TensorVar("c", fmt="d", dtype=torch.float32)
    assign = TensorAssign(c[i], CINBinaryOp(Operation.ADD, a[i], b[i]))
    expect_code("mixed_dtype", ForAll(i, assign))


def test_reduction_without_update_fails_closed():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    y = TensorVar("y", fmt="d")
    assign = TensorAssign(y[i], a[i, j])
    expect_code("unsupported_reduction_without_update", ForAll(i, ForAll(j, assign)))


def test_unsupported_loop_order_kij():
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=Operation.ADD
    )
    expect_code("unsupported_loop_order", ForAll(k, ForAll(i, ForAll(j, assign))))


def test_unsupported_repeated_operand():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.MUL, a[i, j], a[i, j]))
    expect_code("unsupported_repeated_operand", ForAll(i, ForAll(j, assign)))


def test_unsupported_repeated_access_index():
    i = IndexVar("i")
    a = TensorVar("A", fmt="dd")
    y = TensorVar("y", fmt="d")
    assign = TensorAssign(y[i], a[i, i])
    expect_code("unsupported_repeated_access_index", ForAll(i, assign))


def test_unsupported_inplace_operand():
    i = IndexVar("i")
    a = TensorVar("a", fmt="d")
    assign = TensorAssign(a[i], a[i], op=Operation.ADD)
    with pytest.raises(LoopIRLoweringError) as error:
        lower_normalized_cin_to_loopir(normalize_cin(assign_nest(i, assign)))
    assert error.value.defect.code == "unsupported_inplace_operand"


def assign_nest(index_var, assign):
    return ForAll(index_var, assign)


def test_malformed_cin_fails_at_the_cin_verifier():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    # j is never bound by a ForAll: the full CIN verifier owns this failure.
    with pytest.raises(VerificationError):
        lower_normalized_cin_to_loopir(ForAll(i, assign))


def test_non_index_stmt_rejected():
    with pytest.raises(TypeError):
        lower_normalized_cin_to_loopir(object())
