"""Pure iteration-domain / merge-lattice analysis coverage.

The analysis consumes normalized CIN plus its verified LoopPlan and returns
an immutable domain table; it never calls the legacy lowerer, never mutates
its inputs, and fails closed with stable codes on every combination outside
the migrated families.
"""

import copy

import pytest
import torch

from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.loop_plan import LoopPlan, verify_loop_plan
from scorch.compiler.loopir.iterdomain import (
    DomainKind,
    IterationDomainError,
    analyze_iteration_domains,
)


def analyzed(cin):
    loop_ids = []
    current = cin
    while isinstance(current, ForAll):
        loop_ids.append(current.index_var.index_id)
        current = current.stmt
    plan = verify_loop_plan(cin, LoopPlan(loop_order=tuple(loop_ids)))
    return analyze_iteration_domains(cin, plan)


def expect_defect(code, cin):
    loop_ids = []
    current = cin
    while isinstance(current, ForAll):
        loop_ids.append(current.index_var.index_id)
        current = current.stmt
    plan = verify_loop_plan(cin, LoopPlan(loop_order=tuple(loop_ids)))
    with pytest.raises(IterationDomainError) as error:
        analyze_iteration_domains(cin, plan)
    assert error.value.defect.code == code, error.value.defect
    return error.value.defect


def spmv_cin():
    i, j = IndexVar("i"), IndexVar("j")
    y = TensorVar("y", fmt="d", dtype=torch.float32)
    A = TensorVar("A", fmt="ds", dtype=torch.float32)
    x = TensorVar("x", fmt="d", dtype=torch.float32)
    return ForAll(i, ForAll(j, TensorAssign(y[i], A[i, j] * x[j], op=Operation.ADD)))


def elementwise_cin(op, fmt_a="ds", fmt_b="ds", fmt_out="ds"):
    i, j = IndexVar("i"), IndexVar("j")
    C = TensorVar("C", fmt=fmt_out, dtype=torch.float32)
    A = TensorVar("A", fmt=fmt_a, dtype=torch.float32)
    B = TensorVar("B", fmt=fmt_b, dtype=torch.float32)
    if op is Operation.ADD:
        expr = A[i, j] + B[i, j]
    elif op is Operation.SUB:
        expr = A[i, j] - B[i, j]
    else:
        expr = A[i, j] * B[i, j]
    return ForAll(i, ForAll(j, TensorAssign(C[i, j], expr)))


def test_spmv_domains():
    table = analyzed(spmv_cin())
    dense_row, sparse_col = table.domains
    assert dense_row.kind is DomainKind.DENSE
    assert dense_row.cursors == ()
    assert sparse_col.kind is DomainKind.SPARSE
    assert len(sparse_col.cursors) == 1
    assert sparse_col.cursors[0].level == 1


def test_union_and_intersection_classification():
    union = analyzed(elementwise_cin(Operation.ADD)).domains[1]
    assert union.kind is DomainKind.UNION
    assert [ref.level for ref in union.cursors] == [1, 1]
    intersection = analyzed(elementwise_cin(Operation.MUL)).domains[1]
    assert intersection.kind is DomainKind.INTERSECTION
    assert len(intersection.cursors) == 2


@pytest.mark.parametrize("operation", (Operation.ADD, Operation.MUL))
def test_repeated_sparse_operand_collapses_to_one_sparse_domain(operation):
    """Pure support analysis never publishes a one-cursor merge."""

    i, j = IndexVar("i"), IndexVar("j")
    C = TensorVar("C", fmt="dd", dtype=torch.float32)
    A = TensorVar("A", fmt="ds", dtype=torch.float32)
    expression = A[i, j] + A[i, j] if operation is Operation.ADD else A[i, j] * A[i, j]
    domain = analyzed(ForAll(i, ForAll(j, TensorAssign(C[i, j], expression)))).domains[
        1
    ]

    assert domain.kind is DomainKind.SPARSE
    assert len(domain.cursors) == 1


def test_cursor_order_is_rhs_occurrence_order():
    cin = elementwise_cin(Operation.ADD)
    assign = cin.stmt.stmt
    first_symbol = assign.rhs.left.tensor.symbol_id
    table = analyzed(cin)
    assert table.domains[1].cursors[0].tensor == first_symbol


def test_dense_beside_intersection_defers_to_the_sparse_side():
    """(A + B) * D keeps the union domain; the dense operand is loadable."""

    i, j = IndexVar("i"), IndexVar("j")
    C = TensorVar("C", fmt="dd", dtype=torch.float32)
    A = TensorVar("A", fmt="ds", dtype=torch.float32)
    B = TensorVar("B", fmt="ds", dtype=torch.float32)
    D = TensorVar("D", fmt="dd", dtype=torch.float32)
    cin = ForAll(i, ForAll(j, TensorAssign(C[i, j], (A[i, j] + B[i, j]) * D[i, j])))
    domain = analyzed(cin).domains[1]
    assert domain.kind is DomainKind.UNION
    assert len(domain.cursors) == 2


def test_broadcast_variable_resolves_dense():
    i, j = IndexVar("i"), IndexVar("j")
    C = TensorVar("C", fmt="dd", dtype=torch.float32)
    x = TensorVar("x", fmt="d", dtype=torch.float32)
    cin = ForAll(i, ForAll(j, TensorAssign(C[i, j], x[i])))
    table = analyzed(cin)
    assert table.domains[1].kind is DomainKind.DENSE


def test_union_with_dense_fails_closed():
    expect_defect(
        "unsupported_union_with_dense",
        elementwise_cin(Operation.ADD, fmt_b="dd", fmt_out="dd"),
    )


def test_union_with_invariant_fails_closed():
    i, j = IndexVar("i"), IndexVar("j")
    C = TensorVar("C", fmt="dd", dtype=torch.float32)
    A = TensorVar("A", fmt="ds", dtype=torch.float32)
    x = TensorVar("x", fmt="d", dtype=torch.float32)
    cin = ForAll(i, ForAll(j, TensorAssign(C[i, j], A[i, j] + x[i])))
    expect_defect("unsupported_union_operand", cin)


def test_sparse_subtraction_fails_closed():
    expect_defect("unsupported_sparse_subtraction", elementwise_cin(Operation.SUB))


def test_nested_merge_fails_closed():
    i, j = IndexVar("i"), IndexVar("j")
    C = TensorVar("C", fmt="ds", dtype=torch.float32)
    A = TensorVar("A", fmt="ds", dtype=torch.float32)
    B = TensorVar("B", fmt="ds", dtype=torch.float32)
    D = TensorVar("D", fmt="ds", dtype=torch.float32)
    cin = ForAll(i, ForAll(j, TensorAssign(C[i, j], (A[i, j] * B[i, j]) + D[i, j])))
    expect_defect("unsupported_nested_merge", cin)


def test_coordinate_level_type_fails_closed():
    expect_defect(
        "unsupported_level_type",
        elementwise_cin(Operation.MUL, fmt_a="oo", fmt_out="dd"),
    )


def test_plan_mismatch_fails_closed():
    cin = spmv_cin()
    loop_ids = [cin.index_var.index_id, cin.stmt.index_var.index_id]
    plan = verify_loop_plan(cin, LoopPlan(loop_order=tuple(loop_ids)))
    reversed_plan = LoopPlan(loop_order=tuple(reversed(loop_ids)))
    with pytest.raises(IterationDomainError) as error:
        analyze_iteration_domains(cin, reversed_plan)
    assert error.value.defect.code == "plan_mismatch"
    del plan


def test_analysis_is_pure_and_deterministic():
    cin = elementwise_cin(Operation.ADD)
    before = repr(cin)
    first = analyzed(cin)
    second = analyzed(cin)
    assert first == second
    assert repr(cin) == before


def test_analysis_rejects_wrong_types():
    with pytest.raises(TypeError):
        analyze_iteration_domains(object(), LoopPlan(loop_order=()))
    cin = spmv_cin()
    with pytest.raises(TypeError):
        analyze_iteration_domains(cin, object())


def test_domain_table_is_immutable():
    table = analyzed(spmv_cin())
    with pytest.raises(Exception):
        table.domains = ()
    snapshot = copy.deepcopy(table)
    assert snapshot == table


def test_analysis_rejects_cyclic_forall_nest():
    """A self-referential ForAll previously looped without bound."""

    i = IndexVar("i")
    result = TensorVar("C", fmt="d", shape=(4,))
    source = TensorVar("A", fmt="d", shape=(4,))
    nest = ForAll(i, TensorAssign(result[i], source[i], op=Operation.ADD))
    nest.stmt = nest

    with pytest.raises(IterationDomainError) as error:
        analyze_iteration_domains(nest, LoopPlan(loop_order=()))

    assert error.value.defect.code == "unsupported_statement"
