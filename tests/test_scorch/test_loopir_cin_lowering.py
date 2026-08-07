"""Normalized-CIN-to-LoopIR lowering: structure, provenance, and boundaries.

The lowering must map the dense elementwise and reduction/matmul families
onto LoopIR with stable identities preserved, produce a verified LoopPlan,
never mutate its input, and fail closed with stable codes on everything
outside the families.
"""

import pytest
import torch

import scorch.compiler.loopir.pipeline as loopir_pipeline_module
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
from scorch.compiler.compile_options import CompileOptions
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
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    execute_shadow,
)
from scorch.compiler.loopir.verifier import verify_program
from scorch.compiler.scheduler import Schedule, Scheduler


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


def _walk_statements(node):
    """Yield every Stmt reachable from one statement tree node."""

    from scorch.compiler.loopir.nodes import Block, Stmt

    if isinstance(node, Block):
        for statement in node.statements:
            yield from _walk_statements(statement)
        return
    if isinstance(node, Stmt):
        yield node
        body = getattr(node, "body", None)
        if body is not None:
            yield from _walk_statements(body)


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
    # Phase 5 opened DENSE/COMPRESSED level compositions, so the recorded
    # unsupported-format boundary moved to COORDINATE levels and to dense
    # value-bearing leaves below compressed structure.
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="oo")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    expect_code("unsupported_format", ForAll(i, ForAll(j, assign)))


def test_dense_leaf_below_compressed_operand_lowers_to_position_load():
    """The mixed dense-leaf operand chain lowers through a physical load.

    The dense value-bearing leaf below compressed structure reads at its
    leaf position: a dense-position spine grounded at the single-cursor
    bound row position, never a coordinate load or a cursor value.
    """

    from scorch.compiler.loopir.nodes import (
        DensePosition,
        PositionLoad,
        PositionValue,
        SparseFor,
    )

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="sd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    result = lower_normalized_cin_to_loopir(normalize_cin(ForAll(i, ForAll(j, assign))))
    outer = result.program.body.statements[0]
    assert type(outer) is SparseFor
    assert outer.cursor.level == 0
    inner = outer.body.statements[0]
    leaf = inner.body.statements[0]
    load = leaf.value
    assert type(load) is PositionLoad
    assert load.tensor == outer.cursor.tensor
    spine = load.position
    assert type(spine) is DensePosition
    assert spine.level == 1
    assert type(spine.parent) is PositionValue
    assert spine.parent.position == outer.position


def test_mode_order_boundaries():
    """Dense layouts are level-mapped; the rest of the space fails closed.

    A permuted all-dense operand is admitted exactly when its physical
    levels are nest-consistent, so the elementwise copy below now reports
    the loop-order conflict its physical storage actually has.  Permuted
    compressed structure and non-permutation orders keep their own stable
    code.  A permuted result is level-mapped too, so an incompatible nest
    reports the same physical storage-order conflict as an input.
    """

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd", mode_order=[1, 0])
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    expect_code("unsupported_loop_order", ForAll(i, ForAll(j, assign)))

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="ds", mode_order=[1, 0])
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    expect_code("unsupported_mode_order", ForAll(i, ForAll(j, assign)))

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    c = TensorVar("C", fmt="dd", mode_order=[1, 0])
    assign = TensorAssign(c[i, j], a[i, j])
    expect_code("unsupported_loop_order", ForAll(i, ForAll(j, assign)))

    # The shared structural boundary owns permutation totality before
    # scheduling or lowering can perform an unsafe lookup; the lowering's
    # own permutation guard stays as defense in depth behind it.
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd", mode_order=[0, 0])
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    with pytest.raises(VerificationError) as error:
        lower_normalized_cin_to_loopir(ForAll(i, ForAll(j, assign)))
    assert "invalid_cin_field" in str(error.value)


def test_permuted_dense_result_lowers_with_level_mapped_modes():
    """Result declarations keep physical levels distinct from logical axes."""

    i, k = IndexVar("i"), IndexVar("k")
    source = TensorVar("A", fmt="dd", mode_order=[1, 0])
    result = TensorVar("C", fmt="dd", mode_order=[1, 0])
    cin = ForAll(
        k,
        ForAll(i, TensorAssign(result[i, k], source[i, k])),
    )

    lowered = lower_normalized_cin_to_loopir(cin)
    verify_program(lowered.program)

    result_decl = next(
        decl for decl in lowered.program.tensors if decl.symbol == lowered.result_symbol
    )
    assert tuple(level.mode for level in result_decl.levels) == (1, 0)


def test_permuted_dense_operand_lowers_with_level_mapped_modes():
    """The ds@dd transposed matmul lowers with physical level declarations.

    The public einsum("ij,kj->ik", ds, dd) constituent carries its dense
    operand as logical B[k, j] over physical mode_order=[1, 0]; the lowered
    declaration must keep the levels in storage order with their logical
    modes attached, and the base program must verify.
    """

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    result = TensorVar("C", fmt="dd")
    sparse = TensorVar("A", fmt="ds")
    transposed = TensorVar("B", fmt="dd", mode_order=[1, 0])
    cin = ForAll(
        i,
        ForAll(
            j,
            ForAll(
                k,
                TensorAssign(
                    result[i, k],
                    sparse[i, j] * transposed[k, j],
                    op=Operation.ADD,
                ),
            ),
        ),
    )

    lowered = lower_normalized_cin_to_loopir(cin)
    verify_program(lowered.program)

    by_name = {decl.name: decl for decl in lowered.program.tensors}
    assert tuple(level.mode for level in by_name["B"].levels) == (1, 0)
    assert tuple(level.mode for level in by_name["A"].levels) == (0, 1)
    assert tuple(level.mode for level in by_name["C"].levels) == (0, 1)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_implicit_reduction_bridges_to_the_explicit_update(dtype):
    """The public op=None reduction lowers exactly as the explicit ADD twin.

    The frontend deliberately leaves ``TensorAssign.op`` unset; legacy
    iteration analysis re-derives the additive update from the
    right-hand-side-only variables.  The CIN-to-LoopIR boundary owns that
    normalization once, so both spellings must produce one canonical
    program.
    """

    def build(op):
        i, j = IndexVar("i"), IndexVar("j")
        a = TensorVar("A", fmt="dd", dtype=dtype)
        y = TensorVar("y", fmt="d", dtype=dtype)
        return ForAll(i, ForAll(j, TensorAssign(y[i], a[i, j], op=op)))

    implicit = lower_normalized_cin_to_loopir(normalize_cin(build(None)))
    explicit = lower_normalized_cin_to_loopir(normalize_cin(build(Operation.ADD)))
    assert canonical_program_dump(implicit.program) == canonical_program_dump(
        explicit.program
    )
    stores = [
        node
        for node in _walk_statements(implicit.program.body)
        if isinstance(node, (Store, StoreReduce))
    ]
    assert len(stores) == 1
    assert isinstance(stores[0], StoreReduce)
    assert stores[0].op is ReduceOp.ADD


def test_implicit_multi_reduction_bridges_once_at_the_boundary():
    """Several right-hand-side-only loops normalize to one ADD update."""

    def build(op):
        i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
        a = TensorVar("A", fmt="ddd")
        y = TensorVar("y", fmt="d")
        return ForAll(i, ForAll(j, ForAll(k, TensorAssign(y[i], a[i, j, k], op=op))))

    implicit = lower_normalized_cin_to_loopir(normalize_cin(build(None)))
    explicit = lower_normalized_cin_to_loopir(normalize_cin(build(Operation.ADD)))
    assert canonical_program_dump(implicit.program) == canonical_program_dump(
        explicit.program
    )


def test_implicit_bridge_cannot_manufacture_elementwise_updates():
    """A non-reduction op=None assignment stays a plain overwrite store."""

    def build(op):
        i, j = IndexVar("i"), IndexVar("j")
        a = TensorVar("A", fmt="dd")
        b = TensorVar("B", fmt="dd")
        c = TensorVar("C", fmt="dd")
        assign = TensorAssign(
            c[i, j], CINBinaryOp(Operation.MUL, a[i, j], b[i, j]), op=op
        )
        return ForAll(i, ForAll(j, assign))

    implicit = lower_normalized_cin_to_loopir(normalize_cin(build(None)))
    stores = [
        node
        for node in _walk_statements(implicit.program.body)
        if isinstance(node, (Store, StoreReduce))
    ]
    assert len(stores) == 1
    assert isinstance(stores[0], Store)
    explicit = lower_normalized_cin_to_loopir(normalize_cin(build(Operation.ADD)))
    assert canonical_program_dump(implicit.program) != canonical_program_dump(
        explicit.program
    )


def test_implicit_bridge_boundary_rejects_unprovable_reductions():
    """Loops the bridge cannot prove additive never reach normalization.

    ``verify_cin`` owns the only ambiguous shape: a bound loop variable
    used by no access.  Locking that failure proves the bridge sees only
    reduction loops with right-hand-side uses.
    """

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="d")
    c = TensorVar("C", fmt="d")
    with pytest.raises(VerificationError) as error:
        lower_normalized_cin_to_loopir(
            normalize_cin(ForAll(i, ForAll(j, TensorAssign(c[i], a[i]))))
        )
    assert "unused_index_binding" in str(error.value)


def test_implicit_bridge_keeps_repeated_operand_rejection():
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    y = TensorVar("y", fmt="d")
    assign = TensorAssign(y[i], CINBinaryOp(Operation.MUL, a[i, j], a[i, j]))
    expect_code("unsupported_repeated_operand", ForAll(i, ForAll(j, assign)))


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


def _cyclic_forall_program():
    i = IndexVar("i")
    program = ForAll(
        i,
        TensorAssign(TensorVar("C", fmt="d")[i], TensorVar("A", fmt="d")[i]),
    )
    program.stmt = program
    return program


def test_direct_lower_rejects_malformed_structure_before_nest_collection():
    program = _cyclic_forall_program()

    with pytest.raises(VerificationError) as error:
        lower_normalized_cin_to_loopir(program)

    assert error.value.diagnostics[0].code == "cyclic_cin_structure"


@pytest.mark.parametrize("entry", ("compile", "execute", "shadow"))
def test_pipeline_entries_preflight_before_scheduler_metadata_and_build(
    entry,
    monkeypatch,
):
    program = _cyclic_forall_program()
    options = CompileOptions.from_environment(
        environ={},
        requested_schedule=Schedule(),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("malformed CIN escaped structural preflight")

    monkeypatch.setattr(Scheduler, "apply_schedule", forbidden)
    monkeypatch.setattr(loopir_pipeline_module, "_bind_runtime_metadata", forbidden)
    monkeypatch.setattr(
        loopir_pipeline_module,
        "_prepare_generated_kernel_build",
        forbidden,
        raising=False,
    )

    with pytest.raises(VerificationError) as error:
        if entry == "compile":
            compile_cin_via_loopir(
                program,
                (4,),
                (((4,), torch.float32),),
                compile_options=options,
            )
        else:
            if entry == "execute":
                execute_cin_via_loopir(
                    program,
                    (4,),
                    compile_options=options,
                )
            else:
                execute_shadow(
                    program,
                    (4,),
                    compile_options=options,
                )

    assert error.value.diagnostics[0].code == "cyclic_cin_structure"


def test_non_index_stmt_rejected():
    with pytest.raises(TypeError):
        lower_normalized_cin_to_loopir(object())


# -- Phase-5 sparse families ---------------------------------------------------

from scorch.compiler.loopir.nodes import (  # noqa: E402
    AppendEntry,
    CursorValue,
    DensePosition,
    FloatConst,
    MergedSparseFor,
    MergeMode,
    PositionValue,
    RootPosition,
    SparseFor,
)


def build_spmv(fmt_a="ds"):
    i, j = IndexVar("i"), IndexVar("j")
    y = TensorVar("y", fmt="d")
    a = TensorVar("A", fmt=fmt_a)
    x = TensorVar("x", fmt="d")
    assign = TensorAssign(
        y[i], CINBinaryOp(Operation.MUL, a[i, j], x[j]), op=Operation.ADD
    )
    return ForAll(i, ForAll(j, assign)), (i, j), (a, x, y)


def build_sparse_elementwise(op, fmt_out):
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt=fmt_out)
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    assign = TensorAssign(c[i, j], CINBinaryOp(op, a[i, j], b[i, j]))
    return ForAll(i, ForAll(j, assign)), (i, j), (a, b, c)


def test_spmv_lowering_structure():
    cin, (i, j), (a, x, y) = build_spmv()
    result = lower_normalized_cin_to_loopir(normalize_cin(cin))
    program = result.program
    outer = program.body.statements[0]
    assert type(outer) is DenseFor
    sparse_loop = outer.body.statements[0]
    assert type(sparse_loop) is SparseFor
    cursor = sparse_loop.cursor
    assert cursor.tensor == a.symbol_id
    assert cursor.level == 1
    parent = cursor.parent
    assert type(parent) is DensePosition
    assert parent.tensor == a.symbol_id and parent.level == 0
    assert type(parent.parent) is RootPosition
    assert parent.coord.index == i.index_id
    leaf = sparse_loop.body.statements[0]
    assert type(leaf) is StoreReduce
    assert type(leaf.value.lhs) is CursorValue
    assert leaf.value.lhs.default is None
    assert result.loop_plan.loop_order == (i.index_id, j.index_id)


def test_spmm_lowering_keeps_dense_inner_loop():
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=Operation.ADD
    )
    cin = ForAll(i, ForAll(k, ForAll(j, assign)))
    result = lower_normalized_cin_to_loopir(normalize_cin(cin))
    outer = result.program.body.statements[0]
    middle = outer.body.statements[0]
    inner = middle.body.statements[0]
    assert type(outer) is DenseFor
    assert type(middle) is SparseFor
    assert type(inner) is DenseFor


def test_union_add_lowering_structure_and_defaults():
    cin, (i, j), (a, b, c) = build_sparse_elementwise(Operation.ADD, "ds")
    result = lower_normalized_cin_to_loopir(normalize_cin(cin))
    outer = result.program.body.statements[0]
    merged = outer.body.statements[0]
    assert type(merged) is MergedSparseFor
    assert merged.mode is MergeMode.UNION
    assert [cursor.tensor for cursor in merged.cursors] == [
        a.symbol_id,
        b.symbol_id,
    ]
    leaf = merged.body.statements[0]
    assert type(leaf) is AppendEntry
    for side in (leaf.value.lhs, leaf.value.rhs):
        assert type(side) is CursorValue
        assert type(side.default) is FloatConst
        assert side.default.value == 0.0


def test_intersection_multiply_lowering_structure():
    cin, (i, j), (a, b, c) = build_sparse_elementwise(Operation.MUL, "dd")
    result = lower_normalized_cin_to_loopir(normalize_cin(cin))
    merged = result.program.body.statements[0].body.statements[0]
    assert type(merged) is MergedSparseFor
    assert merged.mode is MergeMode.INTERSECTION
    leaf = merged.body.statements[0]
    assert type(leaf) is Store
    for side in (leaf.value.lhs, leaf.value.rhs):
        assert type(side) is CursorValue
        assert side.default is None


def test_dcsr_lowering_links_compressed_parent_positions():
    cin, (i, j), (a, x, y) = build_spmv(fmt_a="ss")
    result = lower_normalized_cin_to_loopir(normalize_cin(cin))
    outer = result.program.body.statements[0]
    assert type(outer) is SparseFor
    assert outer.cursor.level == 0
    assert type(outer.cursor.parent) is RootPosition
    inner = outer.body.statements[0]
    assert type(inner) is SparseFor
    assert type(inner.cursor.parent) is PositionValue
    assert inner.cursor.parent.position == outer.position


def test_sparse_lowering_never_mutates_input():
    cin, _, _ = build_sparse_elementwise(Operation.ADD, "ds")
    normalized = normalize_cin(cin)
    before = canonical_cin_dump(normalized)
    lower_normalized_cin_to_loopir(normalized)
    assert canonical_cin_dump(normalized) == before


def test_sparse_lowering_is_deterministic():
    cin, _, _ = build_sparse_elementwise(Operation.ADD, "ds")
    first = canonical_program_dump(
        lower_normalized_cin_to_loopir(normalize_cin(cin)).program
    )
    second = canonical_program_dump(
        lower_normalized_cin_to_loopir(normalize_cin(cin)).program
    )
    assert first == second


def test_unsupported_sparse_output_layout():
    """A compressed leaf below several dense parents keeps the layout seam.

    Two earlier occupants of this lock have since been migrated: the
    dense-prefix/multi-compressed copy became the single-cursor assembly
    family, and the rank-1 compressed output became the degenerate
    ordered-stream family (``test_loopir_rank1_assembly_target.py``).  The
    seam itself still holds for layouts whose compressed suffix is a
    single level under two or more dense parents, which the assembly
    target's one-dense-prefix rule does not cover.
    """

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dds")
    a = TensorVar("A", fmt="dds")
    assign = TensorAssign(c[i, j, k], a[i, j, k])
    expect_code("unsupported_sparse_output", ForAll(i, ForAll(j, ForAll(k, assign))))


def test_unsupported_sparse_output_reduction():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="ds")
    a = TensorVar("A", fmt="dss")
    assign = TensorAssign(c[i, j], a[i, j, k], op=Operation.ADD)
    expect_code(
        "unsupported_sparse_output_reduction",
        ForAll(i, ForAll(j, ForAll(k, assign))),
    )


def test_unsupported_sparse_output_domain_dense_column():
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="ds")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]))
    expect_code("unsupported_sparse_output_domain", ForAll(i, ForAll(j, assign)))


def test_unsupported_sparse_output_domain_sparse_row():
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="ds")
    a = TensorVar("A", fmt="ss")
    assign = TensorAssign(c[i, j], a[i, j])
    expect_code("unsupported_sparse_output_domain", ForAll(i, ForAll(j, assign)))


def test_unsupported_merged_reduction():
    i, j = IndexVar("i"), IndexVar("j")
    y = TensorVar("y", fmt="d")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    assign = TensorAssign(
        y[i], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]), op=Operation.ADD
    )
    expect_code("unsupported_merged_reduction", ForAll(i, ForAll(j, assign)))


def test_unsupported_merged_update():
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]), op=Operation.ADD
    )
    expect_code("unsupported_merged_update", ForAll(i, ForAll(j, assign)))


def test_analysis_codes_surface_through_the_lowering():
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="ds")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    sub = TensorAssign(c[i, j], CINBinaryOp(Operation.SUB, a[i, j], b[i, j]))
    expect_code("unsupported_sparse_subtraction", ForAll(i, ForAll(j, sub)))

    c2 = TensorVar("C2", fmt="dd")
    dense_b = TensorVar("B2", fmt="dd")
    mixed = TensorAssign(c2[i, j], CINBinaryOp(Operation.ADD, a[i, j], dense_b[i, j]))
    expect_code("unsupported_union_with_dense", ForAll(i, ForAll(j, mixed)))


def test_permuted_layout_canonical_identity_is_distinct():
    """Permuted and identity layouts must never share a canonical identity.

    The canonical program dump serializes each level's stored logical mode,
    so the same logical program over different physical layouts is a
    different artifact; collapsing them would poison schedule and kernel
    caches.
    """

    def build(mode_order):
        i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
        result = TensorVar("C", fmt="dd")
        sparse = TensorVar("A", fmt="ds")
        dense = TensorVar("B", fmt="dd")
        if mode_order is not None:
            dense.mode_order = list(mode_order)
        access = dense[k, j] if mode_order is not None else dense[j, k]
        return ForAll(
            i,
            ForAll(
                j,
                ForAll(
                    k,
                    TensorAssign(result[i, k], sparse[i, j] * access, op=Operation.ADD),
                ),
            ),
        )

    identity_dump = canonical_program_dump(
        lower_normalized_cin_to_loopir(build(None)).program
    )
    permuted_dump = canonical_program_dump(
        lower_normalized_cin_to_loopir(build((1, 0))).program
    )
    assert identity_dump != permuted_dump
    assert '"mode":1' in permuted_dump.replace(" ", "")
