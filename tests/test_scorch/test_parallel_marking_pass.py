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
