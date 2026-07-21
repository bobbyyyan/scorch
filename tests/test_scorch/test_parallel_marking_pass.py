"""Direct contract tests for the parallel-marking/zero-fill pass module."""

from typing import cast

import pytest

from scorch.compiler import llir  # type: ignore[import-untyped]
from scorch.compiler.codegen import LLIRLowerer  # type: ignore[import-untyped]
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    CompilerInvariantError,
)
from scorch.compiler.llir_traversal import (  # type: ignore[import-untyped]
    LLIRRewriter,
    LLIRTraversalContext,
    LLIRWalker,
)
from scorch.compiler.parallel_marking_pass import (  # type: ignore[import-untyped]
    EMPTY_PARALLEL_WORKSPACE_CLUSTER,
    ParallelWorkspaceCluster,
    ParallelWorkspacePoolSpec,
    atomic_work_stealing_prelude,
    mark_first_for_loop_parallel,
)


def _plain_dense_loop() -> llir.ForLoop:
    return llir.ForLoop(
        init=llir.VarInit(
            llir.Var(name="i", type=llir.DataType.INT64),
            llir.Literal(0, llir.DataType.INT),
        ),
        cond=llir.BinOp(
            "<",
            llir.Var(name="i", type=llir.DataType.INT64),
            llir.Var(name="A0_size", type=llir.DataType.INT64),
        ),
        update=llir.Increment(llir.Var(name="i", type=llir.DataType.INT64)),
        body=[],
    )


def _sparse_inner_loop() -> llir.ForLoop:
    return llir.ForLoop(
        init=llir.VarInit(
            llir.Var(name="pA1", type=llir.DataType.INT),
            llir.ArrayAccess(
                array=llir.Var(name="A1_pos", type=llir.DataType.PTR_INT),
                index=llir.Var(name="i", type=llir.DataType.INT),
            ),
        ),
        cond=llir.BinOp(
            "<",
            llir.Var(name="pA1", type=llir.DataType.INT),
            llir.Var(name="pA1_end", type=llir.DataType.INT),
        ),
        update=llir.Increment(llir.Var(name="pA1", type=llir.DataType.INT)),
        body=[],
    )


def _atomic_candidate_loop() -> llir.ForLoop:
    loop = _plain_dense_loop()
    loop.body = [
        llir.VarInit(
            llir.Var(name="pA1_end", type=llir.DataType.INT),
            llir.ArrayAccess(
                array=llir.Var(name="A1_pos", type=llir.DataType.PTR_INT),
                index=llir.Add(
                    llir.Var(name="i", type=llir.DataType.INT),
                    llir.Literal(1, llir.DataType.INT),
                ),
            ),
        ),
        _sparse_inner_loop(),
    ]
    return loop


def _alloc_statement() -> llir.Stmt:
    return llir.VarInit(
        var=llir.Var(
            name="wksp",
            type=llir.DataType.PTR_FLOAT32,
            is_restrict=True,
        ),
        value=llir.MemberCall(
            base=llir.Var(name="wksp_pool_owner", type=llir.DataType.NO_TYPE),
            member="get",
        ),
    )


def _pool_spec() -> ParallelWorkspacePoolSpec:
    return ParallelWorkspacePoolSpec(
        name="wksp",
        scalar_type=llir.DataType.FLOAT32,
        extent="B1_size",
    )


def test_pool_spec_and_cluster_validation_fail_closed() -> None:
    with pytest.raises(TypeError, match="name"):
        ParallelWorkspacePoolSpec(
            name="not an identifier",
            scalar_type=llir.DataType.FLOAT32,
            extent="B1_size",
        )
    with pytest.raises(TypeError, match="scalar_type"):
        ParallelWorkspacePoolSpec(
            name="wksp",
            scalar_type="float",  # type: ignore[arg-type]
            extent="B1_size",
        )
    with pytest.raises(TypeError, match="extent"):
        ParallelWorkspacePoolSpec(
            name="wksp",
            scalar_type=llir.DataType.FLOAT32,
            extent="B1_size + 1",
        )

    with pytest.raises(TypeError, match="alloc"):
        ParallelWorkspaceCluster(alloc=[_alloc_statement()])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="alloc"):
        ParallelWorkspaceCluster(
            alloc=(llir.Literal(0, llir.DataType.INT),)  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="free"):
        ParallelWorkspaceCluster(free=("free();",))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="pool_specs"):
        ParallelWorkspaceCluster(
            pool_specs=(("wksp", llir.DataType.FLOAT32, "B1_size"),)  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="ParallelWorkspaceCluster"):
        mark_first_for_loop_parallel([_plain_dense_loop()], cluster=None)  # type: ignore[arg-type]


def test_empty_cluster_marks_a_plain_parallel_loop() -> None:
    loop = _plain_dense_loop()

    mark_first_for_loop_parallel([loop], EMPTY_PARALLEL_WORKSPACE_CLUSTER)

    assert loop.omp_parallel_for is True
    assert getattr(loop, "_use_atomic_scheduling", False) is False
    assert loop.pre_parallel_body is None
    assert loop.post_parallel_body is None
    assert loop.before_parallel_body is None
    assert loop.omp_num_threads == "scorch_nthreads(-1, A0_size)"
    assert loop.omp_chunk_expr == "scorch_chunk(A0_size, -1)"


def test_cluster_alloc_is_placed_on_the_plain_parallel_path() -> None:
    loop = _plain_dense_loop()
    alloc = _alloc_statement()
    cluster = ParallelWorkspaceCluster(
        alloc=(alloc,),
        pool_specs=(_pool_spec(),),
    )

    mark_first_for_loop_parallel([loop], cluster)

    assert loop.omp_parallel_for is True
    assert loop.pre_parallel_body is not None
    assert alloc in loop.pre_parallel_body
    assert loop.post_parallel_body is None
    assert loop.before_parallel_body is not None
    assert len(loop.before_parallel_body) == 2
    count_init = cast(llir.VarInit, loop.before_parallel_body[0])
    assert count_init.var.name == "wksp_thread_count"
    assert LLIRLowerer().lower_llir(cast(llir.Expr, count_init.value)) == (
        loop.omp_num_threads
    )
    owner_init = cast(llir.VarInit, loop.before_parallel_body[1])
    owner_value = cast(llir.FunctionCall, owner_init.value)
    assert owner_value.template_args == (llir.DataType.FLOAT32,)


def test_atomic_path_owns_prelude_markers_and_typed_policy() -> None:
    loop = _atomic_candidate_loop()
    alloc = _alloc_statement()
    cluster = ParallelWorkspaceCluster(
        alloc=(alloc,),
        pool_specs=(_pool_spec(),),
    )

    mark_first_for_loop_parallel([loop], cluster)

    assert loop.omp_parallel_for is False
    assert getattr(loop, "_use_atomic_scheduling", False) is True
    assert loop._atomic_chunk_var == "_chunk"
    assert loop._atomic_counter_var == "_next_row"
    assert loop._loop_bound == "A0_size"
    assert loop.omp_num_threads == "scorch_nthreads(A1_pos[A0_size], A0_size)"
    assert loop.pre_parallel_body is not None
    assert loop.pre_parallel_body[0] is alloc
    nnz_init, chunk_init = loop.pre_parallel_body[1:]
    assert (
        nnz_init
        == atomic_work_stealing_prelude(
            "A1_pos",
            llir.Var(name="A0_size", type=llir.DataType.INT64),
        )[0]
    )
    assert LLIRLowerer().lower_llir(chunk_init) == (
        "int _chunk = std::max(16, std::min(256, "
        "_nnz / (omp_get_num_threads() * 128)));"
    )
    assert loop.before_parallel_body is not None
    count_init = cast(llir.VarInit, loop.before_parallel_body[0])
    assert LLIRLowerer().lower_llir(cast(llir.Expr, count_init.value)) == (
        loop.omp_num_threads
    )


def test_only_the_first_compatible_loop_is_marked() -> None:
    incompatible = llir.WhileLoop(
        cond=llir.Var(name="go", type=llir.DataType.BOOL),
        body=[],
    )
    first = _plain_dense_loop()
    second = _plain_dense_loop()

    mark_first_for_loop_parallel(
        [incompatible, first, second],
        EMPTY_PARALLEL_WORKSPACE_CLUSTER,
    )

    assert first.omp_parallel_for is True
    assert second.omp_parallel_for is False


def test_placed_statements_survive_traversal_and_detachment() -> None:
    loop = _atomic_candidate_loop()
    cluster = ParallelWorkspaceCluster(
        alloc=(_alloc_statement(),),
        pool_specs=(_pool_spec(),),
    )
    mark_first_for_loop_parallel([loop], cluster)

    context = LLIRTraversalContext(
        stage="parallel marking test",
        pass_name="detach",
    )
    assert loop.pre_parallel_body is not None
    assert loop.before_parallel_body is not None
    for statement in (*loop.pre_parallel_body, *loop.before_parallel_body):
        LLIRWalker(context).walk(statement)
        detached = LLIRRewriter(context).rewrite(statement)
        assert detached == statement
        assert detached is not statement


def test_pool_attachment_without_typed_policy_fails_closed() -> None:
    # A compatible `<=` loop hides its bound from the policy helpers, so a
    # preexisting pragma string has no typed twin and the pool must refuse.
    loop = _plain_dense_loop()
    loop.cond = llir.BinOp(
        "<=",
        llir.Var(name="i", type=llir.DataType.INT64),
        llir.Var(name="A0_size", type=llir.DataType.INT64),
    )
    loop.omp_num_threads = "scorch_nthreads(opaque, rows)"
    cluster = ParallelWorkspaceCluster(pool_specs=(_pool_spec(),))

    with pytest.raises(CompilerInvariantError, match="no typed value"):
        mark_first_for_loop_parallel([loop], cluster)
