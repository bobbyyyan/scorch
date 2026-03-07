from scorch.compiler.cin import (
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
)
from scorch.compiler.scheduler import Scheduler


def _build_spmm_cin(fmt_c: str, fmt_a: str, fmt_b: str, loop_order: str) -> ForAll:
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    ivars = {"i": i, "j": j, "k": k}

    C = TensorVar("C", fmt=fmt_c)
    A = TensorVar("A", fmt=fmt_a)
    B = TensorVar("B", fmt=fmt_b)

    assign = TensorAssign(C[i, k], A[i, j] * B[j, k], op=Operation.ADD)

    stmt = assign
    for ch in reversed(loop_order):
        stmt = ForAll(ivars[ch], stmt)

    assert isinstance(stmt, ForAll)
    return stmt


def _loop_names(stmt):
    names = []
    curr = stmt
    while isinstance(curr, ForAll):
        names.append(curr.index_var.name)
        curr = curr.stmt
    return names, curr


def test_sort_by_sparsity_descending_spmm():
    stmt = _build_spmm_cin(fmt_c="dd", fmt_a="ds", fmt_b="dd", loop_order="ijk")
    index_vars = Scheduler.get_index_variables(stmt)
    sorted_vars = Scheduler.sort_by_sparsity_descending(index_vars, stmt)
    assert [v.name for v in sorted_vars] == ["j", "i", "k"]


def test_cost_to_push_spmm_prefers_gustavson_over_inner_product():
    stmt = _build_spmm_cin(fmt_c="dd", fmt_a="ds", fmt_b="dd", loop_order="ijk")
    init_order = Scheduler.init_loop_order(stmt)  # [j, i, k]
    j = init_order[0]

    # Moving j from outermost to Gustavson position is beneficial
    cost_to_gustavson = Scheduler.cost_to_push(stmt, init_order, j, 1)
    assert cost_to_gustavson < 0

    # Once at Gustavson, moving j further to inner product is not beneficial
    # (dense output + dense B = no workspace or transposition penalty either way)
    gustavson_order = Scheduler.move_to_position(init_order, j, 1)  # [i, j, k]
    cost_to_inner = Scheduler.cost_to_push(stmt, gustavson_order, j, 2)
    assert cost_to_inner >= 0


def test_optimize_loop_order_spmm():
    stmt = _build_spmm_cin(fmt_c="dd", fmt_a="ds", fmt_b="dd", loop_order="ijk")
    init_order = Scheduler.init_loop_order(stmt)
    optimized_order = Scheduler.optimize_loop_order(stmt, init_order)
    assert [v.name for v in optimized_order] == ["i", "j", "k"]


def test_apply_mode_order_constraints_breaks_cycle():
    i = IndexVar("i")
    j = IndexVar("j")

    C = TensorVar("C", fmt="dd", mode_order=[0, 1])
    A = TensorVar("A", fmt="dd", mode_order=[0, 1])
    B = TensorVar("B", fmt="dd", mode_order=[1, 0])

    stmt = ForAll(
        j,
        ForAll(
            i,
            TensorAssign(C[i, j], A[i, j] * B[j, i], op=Operation.ADD),
        ),
    )

    constrained = Scheduler.apply_mode_order_constraints(stmt, [j, i])
    assert [v.name for v in constrained] == ["i", "j"]


def test_should_insert_workspace_depends_on_reduction_position():
    i = IndexVar("i")
    j = IndexVar("j")
    k = IndexVar("k")
    C = TensorVar("C", fmt="ds")
    A = TensorVar("A", fmt="ds")
    B = TensorVar("B", fmt="dd")
    stmt = ForAll(
        i,
        ForAll(
            j,
            ForAll(k, TensorAssign(C[i, k], A[i, j] * B[j, k], op=Operation.ADD)),
        ),
    )

    assert Scheduler.should_insert_workspace(stmt, [i, j, k])
    assert not Scheduler.should_insert_workspace(stmt, [i, k, j])


def test_auto_schedule_spmspm_inserts_workspace():
    stmt = _build_spmm_cin(fmt_c="ds", fmt_a="ds", fmt_b="ds", loop_order="ijk")
    scheduled = Scheduler.auto_schedule(stmt)

    outer_loop_names, body = _loop_names(scheduled)
    assert outer_loop_names == ["i"]
    assert isinstance(body, Where)
    assert scheduled.inserted_workspace

    producer_loop_names, _ = _loop_names(body.producer)
    consumer_loop_names, _ = _loop_names(body.consumer)
    assert producer_loop_names == ["j", "k"]
    assert consumer_loop_names == ["j", "k"]


def test_auto_schedule_spmm_dense_output_no_workspace():
    stmt = _build_spmm_cin(fmt_c="dd", fmt_a="ds", fmt_b="dd", loop_order="ijk")
    scheduled = Scheduler.auto_schedule(stmt)

    outer_loop_names, body = _loop_names(scheduled)
    assert not isinstance(body, Where)
    assert not scheduled.inserted_workspace
