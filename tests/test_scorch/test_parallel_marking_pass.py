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
    POOL_COO_WORKSPACE_1D,
    ParallelWorkspaceCluster,
    ParallelWorkspacePoolSpec,
    atomic_work_stealing_prelude,
    mark_first_for_loop_parallel,
    scalars_carried_between_iterations,
    serial_workspace_cluster_placement,
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


def _free_statement() -> llir.Stmt:
    return llir.MemberCallStmt(
        base=llir.Var(name="wksp", type=llir.DataType.PTR_FLOAT32),
        member="release",
    )


def _mutable_llir_ids(value: object) -> set[int]:
    mutable_ids: set[int] = set()
    if isinstance(value, llir.Node):
        mutable_ids.add(id(value))
        for child in vars(value).values():
            mutable_ids.update(_mutable_llir_ids(child))
    elif type(value) is ParallelWorkspaceCluster:
        for child in vars(value).values():
            mutable_ids.update(_mutable_llir_ids(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            mutable_ids.update(_mutable_llir_ids(child))
    return mutable_ids


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
    for unsupported_scalar in (
        llir.DataType.NO_TYPE,
        llir.DataType.PTR_FLOAT32,
        llir.DataType.STD_VECTOR_INT,
        llir.DataType.TORCH_FLOAT32,
    ):
        with pytest.raises(TypeError, match="supported scalar DataType"):
            ParallelWorkspacePoolSpec(
                name="wksp",
                scalar_type=unsupported_scalar,
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

    forged_spec = _pool_spec()
    object.__setattr__(forged_spec, "scalar_type", llir.DataType.NO_TYPE)
    forged_cluster = ParallelWorkspaceCluster()
    object.__setattr__(forged_cluster, "pool_specs", (forged_spec,))
    with pytest.raises(TypeError, match="supported scalar DataType"):
        mark_first_for_loop_parallel([_plain_dense_loop()], forged_cluster)

    missing_cluster_field = ParallelWorkspaceCluster()
    del missing_cluster_field.__dict__["alloc"]
    with pytest.raises(TypeError, match="alloc"):
        mark_first_for_loop_parallel([_plain_dense_loop()], missing_cluster_field)

    class UnknownStatement(llir.Stmt):
        pass

    with pytest.raises(TypeError, match="alloc"):
        ParallelWorkspaceCluster(alloc=(UnknownStatement(),))
    with pytest.raises(TypeError, match="alloc"):
        ParallelWorkspaceCluster(alloc=(llir.RawStmt("raw();"),))

    malformed_alloc = cast(llir.VarInit, _alloc_statement())
    del malformed_alloc.__dict__["value"]
    with pytest.raises(TypeError, match="complete plain VarInit"):
        ParallelWorkspaceCluster(alloc=(malformed_alloc,))

    for field, invalid_value in (("op", "; injected"), ("cast", True)):
        malformed_alloc = cast(llir.VarInit, _alloc_statement())
        setattr(malformed_alloc, field, invalid_value)
        with pytest.raises(TypeError, match="complete plain VarInit"):
            ParallelWorkspaceCluster(alloc=(malformed_alloc,))

    cyclic_value = llir.Add(
        llir.Var(name="lhs", type=llir.DataType.INT64),
        llir.Var(name="rhs", type=llir.DataType.INT64),
    )
    object.__setattr__(cyclic_value, "left", cyclic_value)
    cyclic_alloc = llir.VarInit(
        var=llir.Var(name="value", type=llir.DataType.INT64),
        value=cyclic_value,
    )
    with pytest.raises(TypeError, match="acyclic"):
        ParallelWorkspaceCluster(alloc=(cyclic_alloc,))

    forged_statement = cast(llir.VarInit, _alloc_statement())
    cluster_with_forged_child = ParallelWorkspaceCluster(alloc=(forged_statement,))
    del forged_statement.__dict__["value"]
    with pytest.raises(TypeError, match="complete plain VarInit"):
        mark_first_for_loop_parallel([_plain_dense_loop()], cluster_with_forged_child)


def test_cluster_templates_reject_unsafe_structured_spellings() -> None:
    safe_target = llir.Var(name="wksp", type=llir.DataType.INT64)
    safe_value = llir.Var(name="source", type=llir.DataType.INT64)
    unsafe_templates = (
        llir.VarInit(
            var=llir.Var(
                name="wksp; injected()",
                type=llir.DataType.INT64,
            ),
            value=safe_value,
        ),
        llir.VarInit(
            var=safe_target,
            value=llir.Var(
                name="source; injected()",
                type=llir.DataType.INT64,
            ),
        ),
        llir.VarInit(
            var=llir.Var(
                name="wksp",
                type="int; injected",  # type: ignore[arg-type]
            ),
            value=safe_value,
        ),
        llir.VarInit(
            var=safe_target,
            value=llir.FunctionCall("f(); injected"),
        ),
        llir.MemberCallStmt(
            base=llir.Var(
                name="wksp; injected()",
                type=llir.DataType.PTR_FLOAT32,
            ),
            member="release",
        ),
        llir.MemberCallStmt(
            base=llir.Var(name="wksp", type=llir.DataType.PTR_FLOAT32),
            member="class",
        ),
    )

    for template in unsafe_templates:
        with pytest.raises(TypeError, match="ParallelWorkspaceCluster"):
            ParallelWorkspaceCluster(alloc=(template,))


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
    # The policy is evaluated once, for the pool's count; every worker slices the
    # pool by omp_get_thread_num(), so the region's team is that count.
    assert LLIRLowerer().lower_llir(cast(llir.Expr, count_init.value)) == (
        "scorch_nthreads(-1, A0_size)"
    )
    assert loop.omp_num_threads == "wksp_thread_count"
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
    assert loop.omp_num_threads == "wksp_thread_count"
    assert loop.pre_parallel_body is not None
    assert loop.pre_parallel_body[0] == alloc
    assert loop.pre_parallel_body[0] is not alloc
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
        "scorch_nthreads(A1_pos[A0_size], A0_size)"
    )


def test_two_pools_give_the_region_the_smaller_count() -> None:
    loop = _plain_dense_loop()
    second = ParallelWorkspacePoolSpec(
        name="other",
        scalar_type=llir.DataType.FLOAT64,
        extent="B1_size",
    )
    cluster = ParallelWorkspaceCluster(pool_specs=(_pool_spec(), second))

    mark_first_for_loop_parallel([loop], cluster)

    assert loop.before_parallel_body is not None
    counts = [
        cast(llir.VarInit, statement)
        for statement in loop.before_parallel_body
        if cast(llir.VarInit, statement).var.name.endswith("_thread_count")
    ]
    assert [count.var.name for count in counts] == [
        "wksp_thread_count",
        "other_thread_count",
    ]
    assert loop.omp_num_threads == "std::min(wksp_thread_count, other_thread_count)"
    assert "num_threads(std::min(wksp_thread_count, other_thread_count))" in (
        LLIRLowerer().lower_llir(loop)
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


def test_cluster_placements_are_independently_owned_per_marking() -> None:
    alloc = _alloc_statement()
    free = _free_statement()
    cluster = ParallelWorkspaceCluster(alloc=(alloc,), free=(free,))
    first = _plain_dense_loop()
    second = _plain_dense_loop()

    mark_first_for_loop_parallel([first], cluster)
    mark_first_for_loop_parallel([second], cluster)

    assert first.pre_parallel_body is not None
    assert first.post_parallel_body is not None
    assert second.pre_parallel_body is not None
    assert second.post_parallel_body is not None
    assert first.pre_parallel_body[0] == second.pre_parallel_body[0] == alloc
    assert first.post_parallel_body[0] == second.post_parallel_body[0] == free
    assert _mutable_llir_ids(cluster).isdisjoint(_mutable_llir_ids(first))
    assert _mutable_llir_ids(cluster).isdisjoint(_mutable_llir_ids(second))
    assert _mutable_llir_ids(first).isdisjoint(_mutable_llir_ids(second))

    first_alloc = cast(llir.VarInit, first.pre_parallel_body[0])
    first_alloc.var.name = "mutated_first_workspace"
    assert cast(llir.VarInit, cluster.alloc[0]).var.name == "wksp"
    assert cast(llir.VarInit, second.pre_parallel_body[0]).var.name == "wksp"


def test_atomic_prelude_validates_its_structural_name_pair() -> None:
    bound = llir.Var(name="A0_size", type=llir.DataType.INT64)

    for sparse_pos in ("A1_pos); injected(", "A1_pos\n", "B1_pos", 17):
        with pytest.raises(TypeError, match="atomic prelude"):
            atomic_work_stealing_prelude(sparse_pos, bound)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="exact Var"):
        atomic_work_stealing_prelude("A1_pos", object())  # type: ignore[arg-type]

    forged_bound = llir.Var(name="A0_size", type=llir.DataType.INT64)
    del forged_bound.__dict__["type"]
    with pytest.raises(TypeError, match="complete fields"):
        atomic_work_stealing_prelude("A1_pos", forged_bound)

    pointer_bound = llir.Var(
        name="A0_size",
        type=llir.DataType.INT64,
        is_ptr=True,
    )
    with pytest.raises(TypeError, match="metadata-free scalar Var"):
        atomic_work_stealing_prelude("A1_pos", pointer_bound)

    for invalid_type in (
        llir.DataType.BOOL,
        llir.DataType.FLOAT32,
        llir.DataType.VOID,
        llir.DataType.NO_TYPE,
        llir.DataType.PTR_FLOAT32,
        llir.DataType.STD_VECTOR_INT,
        llir.DataType.TORCH_TENSOR,
    ):
        invalid_bound = llir.Var(name="A0_size", type=invalid_type)
        with pytest.raises(TypeError, match="metadata-free scalar Var"):
            atomic_work_stealing_prelude("A1_pos", invalid_bound)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("name", "class"),
        ("name", "wksp\n"),
        ("extent", "true"),
        ("extent", "and"),
    ],
)
def test_pool_spec_rejects_non_cpp_identifiers(field: str, invalid_value: str) -> None:
    fields = {
        "name": "wksp",
        "scalar_type": llir.DataType.FLOAT32,
        "extent": "wksp_size",
    }
    fields[field] = invalid_value

    with pytest.raises(TypeError, match=rf"ParallelWorkspacePoolSpec\.{field}"):
        ParallelWorkspacePoolSpec(**fields)

    valid_spec = _pool_spec()
    object.__setattr__(valid_spec, field, invalid_value)
    forged_cluster = ParallelWorkspaceCluster()
    object.__setattr__(forged_cluster, "pool_specs", (valid_spec,))
    with pytest.raises(TypeError, match=rf"ParallelWorkspacePoolSpec\.{field}"):
        mark_first_for_loop_parallel([_plain_dense_loop()], forged_cluster)


def test_pool_spec_preserves_unicode_identifier_compatibility() -> None:
    assert (
        ParallelWorkspacePoolSpec(
            name="dénse",
            scalar_type=llir.DataType.FLOAT32,
            extent="éxtent",
        ).extent
        == "éxtent"
    )


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


def _int(name: str) -> llir.Var:
    return llir.Var(name=name, type=llir.DataType.INT)


def _cursor_advancing_loop() -> llir.ForLoop:
    """``for (i ...) { int i_C = C0_crd[pC0]; pC0 += (int)(i_C == i); }``.

    The loop a sum of a dense and a compressed operand emits over the dense
    extent: ``pC0`` is declared before it and advanced at the end of each
    iteration.
    """

    loop = _plain_dense_loop()
    loop.body = [
        llir.VarInit(
            _int("i_C"),
            llir.ArrayAccess(
                array=llir.Var(name="C0_crd", type=llir.DataType.PTR_INT),
                index=_int("pC0"),
            ),
        ),
        llir.Assign(
            _int("pC0"),
            llir.BinOp("==", _int("i_C"), _int("i")),
            llir.AssignOp.ADD_ASSIGN,
        ),
    ]
    return loop


def test_a_loop_that_advances_an_outer_cursor_is_not_marked() -> None:
    loop = _cursor_advancing_loop()

    marked = mark_first_for_loop_parallel([loop], EMPTY_PARALLEL_WORKSPACE_CLUSTER)

    assert scalars_carried_between_iterations(loop) == ("pC0",)
    assert marked is False
    assert loop.omp_parallel_for is False
    assert loop.omp_num_threads is None
    assert "#pragma omp" not in LLIRLowerer().lower_llir(loop)


def test_a_carrying_loop_ends_the_search_and_no_later_loop_is_marked() -> None:
    carrying = _cursor_advancing_loop()
    independent = _plain_dense_loop()

    marked = mark_first_for_loop_parallel(
        [carrying, independent], EMPTY_PARALLEL_WORKSPACE_CLUSTER
    )

    assert marked is False
    assert carrying.omp_parallel_for is False
    assert independent.omp_parallel_for is False


def test_a_cluster_is_not_placed_when_the_first_loop_carries_a_scalar() -> None:
    carrying = _cursor_advancing_loop()
    independent = _plain_dense_loop()
    cluster = ParallelWorkspaceCluster(
        alloc=(_alloc_statement(),),
        pool_specs=(_pool_spec(),),
    )

    assert mark_first_for_loop_parallel([carrying, independent], cluster) is False
    for loop in (carrying, independent):
        assert loop.pre_parallel_body is None
        assert loop.before_parallel_body is None


def test_carried_scalars_are_found_at_any_depth_in_first_assignment_order() -> None:
    loop = _plain_dense_loop()
    loop.body = [
        llir.IfThenElse(
            cond=llir.BinOp("<", _int("pC0"), _int("pC0_end")),
            then_body=[
                llir.WhileLoop(
                    cond=llir.BinOp("<", _int("pC1_end"), _int("pC0_end")),
                    body=[llir.Increment(_int("pC1_end"))],
                )
            ],
            else_body=[llir.Assign(_int("pC0"), _int("pC1_end"))],
        ),
        llir.Increment(_int("pC1_end")),
    ]

    assert scalars_carried_between_iterations(loop) == ("pC1_end", "pC0")


def test_a_scalar_the_body_declares_is_not_carried() -> None:
    loop = _plain_dense_loop()
    inner = _sparse_inner_loop()
    inner.body = [llir.Increment(_int("count"))]
    loop.body = [
        llir.VarInit(_int("count"), llir.Literal(0, llir.DataType.INT)),
        llir.VarDecl(_int("last")),
        inner,
        llir.Assign(_int("last"), _int("count")),
    ]

    assert scalars_carried_between_iterations(loop) == ()
    assert mark_first_for_loop_parallel([loop], EMPTY_PARALLEL_WORKSPACE_CLUSTER)
    assert loop.omp_parallel_for is True


def test_a_store_through_a_subscript_is_not_a_carried_scalar() -> None:
    loop = _plain_dense_loop()
    loop.body = [
        llir.Assign(
            llir.ArrayAccess(
                array=llir.Var(name="A_values", type=llir.DataType.PTR_FLOAT32),
                index=llir.Var(name="i", type=llir.DataType.INT64),
            ),
            llir.Literal(0, llir.DataType.INT),
        )
    ]

    assert scalars_carried_between_iterations(loop) == ()


def test_a_scalar_declared_inside_the_parallel_region_is_one_per_worker() -> None:
    loop = _plain_dense_loop()
    loop.body = [llir.Increment(_int("wksp_count"))]
    per_worker = [llir.VarInit(_int("wksp_count"), llir.Literal(0, llir.DataType.INT))]

    assert scalars_carried_between_iterations(loop) == ("wksp_count",)
    assert scalars_carried_between_iterations(loop, per_worker) == ()


def _coo_pool_spec() -> ParallelWorkspacePoolSpec:
    return ParallelWorkspacePoolSpec(
        name="w",
        scalar_type=llir.DataType.FLOAT64,
        extent="C1_size",
        container=POOL_COO_WORKSPACE_1D,
    )


def _coo_borrow(array: llir.Expr) -> llir.Stmt:
    return llir.VarInit(
        var=llir.Var(name="w", type=llir.DataType.AUTO),
        value=llir.MemberCall(
            base=llir.ArrayAccess(
                array=array,
                index=llir.Cast(
                    expr=llir.FunctionCall("omp_get_thread_num"),
                    data_type=llir.DataType.SIZE_T,
                ),
            ),
            member="make_region_view",
        ),
    )


def _owner() -> llir.Var:
    return llir.Var(name="w_pool_owner", type=llir.DataType.NO_TYPE)


def _render(statements: list) -> str:
    return "\n".join(LLIRLowerer().lower_llir(s).strip() for s in statements)


def _row_store() -> llir.Stmt:
    """``out[i] = 1.0``: each iteration writes its own element, so nothing is
    carried from one iteration to the next."""

    return llir.Assign(
        llir.ArrayAccess(
            array=llir.Var(name="out", type=llir.DataType.PTR_FLOAT64),
            index=llir.Var(name="i", type=llir.DataType.INT64),
        ),
        llir.Literal(1.0, llir.DataType.FLOAT64),
    )


def test_a_sparse_workspace_pool_is_built_before_the_region() -> None:
    """One ``coo_workspace_owner_1d`` per worker, of capacity ``extent``, built
    before the parallel region with nothing allocated; inside it each worker
    takes its view by ``make_region_view()``, skips its rows if that view
    failed, and the statement after the loop checks the pool."""

    loop = _plain_dense_loop()
    loop.body = [_row_store()]
    statements: list = [loop]
    cluster = ParallelWorkspaceCluster(
        alloc=(_coo_borrow(_owner()),), pool_specs=(_coo_pool_spec(),)
    )
    assert mark_first_for_loop_parallel(statements, cluster)
    before = _render(loop.before_parallel_body)
    assert before.index("int w_thread_count = ") < before.index(
        "std::vector<coo_workspace_owner_1d<double>> w_pool_owner;"
    )
    assert "w_pool_owner.reserve((size_t)w_thread_count);" in before
    assert "for (int _worker = 0; _worker < w_thread_count; _worker++)" in before
    assert "w_pool_owner.emplace_back(C1_size, true);" in before
    assert "scorch_make_aligned_buffer" not in before
    assert loop.omp_num_threads == "w_thread_count"
    assert _render(loop.pre_parallel_body) == (
        "auto w = w_pool_owner[(size_t)omp_get_thread_num()].make_region_view();"
    )
    skip = loop.body[0]
    assert type(skip) is llir.IfThenElse
    assert " ".join(_render([skip]).split()) == "if (w.failed()) { continue; }"
    assert type(loop.body[1]) is llir.Assign
    assert len(statements) == 2 and statements[0] is loop
    assert _render(statements[1:]) == "scorch_check_workspace_pool(w_pool_owner);"


def test_a_loop_carrying_a_scalar_gets_no_pool_skip_or_check() -> None:
    """A row loop whose body advances a scalar from one iteration to the next
    (``rows++``) must run in order, so the pass leaves it unmarked: no pool, no
    skip in its body and no check after it (the caller places the cluster
    serially)."""

    loop = _plain_dense_loop()
    loop.body = [llir.Increment(_int("rows"))]
    statements: list = [loop]
    cluster = ParallelWorkspaceCluster(
        alloc=(_coo_borrow(_owner()),), pool_specs=(_coo_pool_spec(),)
    )
    assert not mark_first_for_loop_parallel(statements, cluster)
    assert statements == [loop]
    assert len(loop.body) == 1 and type(loop.body[0]) is llir.Increment
    assert not loop.omp_parallel_for
    assert not loop.before_parallel_body
    assert not loop.pre_parallel_body


def test_the_pool_check_needs_the_loops_statement_list() -> None:
    """The check after the region is placed in the loop's own statement list;
    a sequence that cannot take it fails closed."""

    cluster = ParallelWorkspaceCluster(
        alloc=(_coo_borrow(_owner()),), pool_specs=(_coo_pool_spec(),)
    )
    with pytest.raises(CompilerInvariantError) as raised:
        mark_first_for_loop_parallel((_plain_dense_loop(),), cluster)
    assert raised.value.code == "workspace_placement_failed"


def test_a_dense_workspace_pool_gets_no_check() -> None:
    loop = _plain_dense_loop()
    statements: list = [loop]
    dense = ParallelWorkspacePoolSpec(
        name="w", scalar_type=llir.DataType.FLOAT64, extent="C1_size"
    )
    assert mark_first_for_loop_parallel(
        statements, ParallelWorkspaceCluster(pool_specs=(dense,))
    )
    assert statements == [loop]
    assert loop.body == []


def test_a_serial_kernel_takes_the_pools_one_view() -> None:
    cluster = ParallelWorkspaceCluster(
        alloc=(_coo_borrow(_owner()),), pool_specs=(_coo_pool_spec(),)
    )
    before, after = serial_workspace_cluster_placement(cluster)
    text = _render(before)
    assert "std::vector<coo_workspace_owner_1d<double>> w_pool_owner;" in text
    assert "w_pool_owner.reserve((size_t)1);" in text
    assert "w_pool_owner.emplace_back(C1_size);" in text
    assert text.endswith("auto w = w_pool_owner[0].make_view();")
    assert "omp_get_thread_num" not in text
    assert "make_region_view" not in text
    assert "scorch_check_workspace_pool" not in text
    assert after == []


def test_the_pool_container_and_the_element_borrow_fail_closed() -> None:
    with pytest.raises(TypeError, match=r"ParallelWorkspacePoolSpec\.container"):
        ParallelWorkspacePoolSpec(
            name="w",
            scalar_type=llir.DataType.FLOAT64,
            extent="C1_size",
            container="hash_map",
        )
    with pytest.raises(TypeError, match="workspace-borrow expression grammar"):
        ParallelWorkspaceCluster(
            alloc=(_coo_borrow(llir.Literal(0, llir.DataType.INT)),)
        )
    forged = _coo_borrow(_owner())
    view_call = cast(llir.MemberCall, cast(llir.VarInit, forged).value)
    object.__setattr__(view_call.base, "tensor_access", object())
    with pytest.raises(TypeError, match="tensor_access must be absent"):
        ParallelWorkspaceCluster(alloc=(forged,))
