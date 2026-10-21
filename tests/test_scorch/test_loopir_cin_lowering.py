"""Normalized-CIN-to-LoopIR lowering: structure, provenance, and boundaries.

The lowering must map the dense elementwise and reduction/matmul families
onto LoopIR with stable identities preserved, produce a verified LoopPlan,
never mutate its input, and fail closed with stable codes on everything
outside the families.
"""

import difflib

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
from scorch.compiler.diagnostics import InvalidSchedule, VerificationError
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
    compare_generated_sources,
    compile_cin_via_loopir,
    execute_cin_via_loopir,
    execute_shadow,
    legacy_generated_cpp,
)
from scorch.compiler.loopir.verifier import verify_program
from scorch.compiler.scheduler import Schedule, Scheduler
from scorch.stensor import STensor


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
    workspace = Workspace(name="wksp", dim=0, dtype=y.dtype)
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


@pytest.mark.parametrize("fmt", ["compressed,singleton", "singleton,singleton"])
def test_unsupported_format(fmt):
    # Phase 5 opened DENSE/COMPRESSED level compositions, and all-coordinate
    # operands and results are lowered, so the recorded unsupported-format
    # boundary is a level type outside those three: a singleton level.
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt=fmt)
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    expect_code("unsupported_format", ForAll(i, ForAll(j, assign)))


def test_an_all_coordinate_result_is_declared_with_coordinate_levels():
    from scorch.compiler.loopir.nodes import AppendEntry, LevelKind

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="oo")
    c = TensorVar("C", fmt="oo")
    assign = TensorAssign(c[i, j], a[i, j])
    program = lower_normalized_cin_to_loopir(
        normalize_cin(ForAll(i, ForAll(j, assign)))
    ).program
    verify_program(program)
    result = next(decl for decl in program.tensors if decl.name == "C")
    assert [level.kind for level in result.levels] == [LevelKind.COORDINATE] * 2
    appends = [
        node for node in _walk_statements(program.body) if type(node) is AppendEntry
    ]
    assert len(appends) == 1 and appends[0].tensor == result.symbol


@pytest.mark.parametrize("fmt", ["do", "od", "so", "os"])
def test_a_coordinate_level_beside_another_kind_is_refused(fmt):
    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt=fmt)
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    expect_code("unsupported_mixed_coordinate_hierarchy", ForAll(i, ForAll(j, assign)))


def test_an_all_coordinate_operand_lowers_to_one_sparse_loop_per_level():
    from scorch.compiler.loopir.nodes import LevelKind, SparseFor

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="oo")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    program = lower_normalized_cin_to_loopir(
        normalize_cin(ForAll(i, ForAll(j, assign)))
    ).program
    loops = [node for node in _walk_statements(program.body) if type(node) is SparseFor]
    assert [loop.cursor.level for loop in loops] == [0, 1]
    operand = next(decl for decl in program.tensors if decl.name == "A")
    assert [level.kind for level in operand.levels] == [LevelKind.COORDINATE] * 2


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


def permuted_dense_operand_copy():
    """``C[i,j] = A[i,j]`` in the nest ``i, j`` with ``A`` stored ``(j, i)``."""

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd", mode_order=[1, 0])
    c = TensorVar("C", fmt="dd")
    return ForAll(i, ForAll(j, TensorAssign(c[i, j], a[i, j])))


def permuted_dense_result_copy():
    """``C[i,j] = A[i,j]`` in the nest ``i, j`` with ``C`` stored ``(j, i)``."""

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    c = TensorVar("C", fmt="dd", mode_order=[1, 0])
    return ForAll(i, ForAll(j, TensorAssign(c[i, j], a[i, j])))


def test_mode_order_boundaries():
    """Dense layouts are level-mapped; the rest of the space fails closed.

    A permuted all-dense operand or result is read or written by its
    coordinates through its level map, in any loop order.  The two copies
    below visit a permuted dense tensor against its physical storage order;
    they reported that conflict (``unsupported_loop_order``) until the
    lowering stopped requiring a storage-order visit of a tensor with no
    compressed level, and now lower.  Permuted compressed structure handed
    straight to this lowering keeps its own stable code (the pipeline rewrites
    such a program into storage-order form before it gets here,
    ``loop_plan.storage_order_form``), and so does a non-permutation order.
    """

    for build, name in (
        (permuted_dense_operand_copy, "A"),
        (permuted_dense_result_copy, "C"),
    ):
        lowered = lower_normalized_cin_to_loopir(normalize_cin(build()))
        verify_program(lowered.program)
        decl = next(decl for decl in lowered.program.tensors if decl.name == name)
        assert tuple(level.mode for level in decl.levels) == (1, 0), name

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="ds", mode_order=[1, 0])
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(c[i, j], a[i, j])
    expect_code("unsupported_mode_order", ForAll(i, ForAll(j, assign)))

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


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["plain", "blocked"])
def test_a_permuted_dense_tensor_visited_against_its_storage_order_compiles(
    regblock_enabled,
):
    """The two copies above, compiled as written, against the shipped compiler.

    A shape is bound per storage level: ``A`` stored ``(j, i)`` is ``(4, 3)``
    for a ``3 x 4`` result, and ``C`` stored ``(j, i)`` is ``(4, 3)`` for a
    ``3 x 4`` operand.  Each position is computed through the level map.  The
    result copy is the shipped compiler's kernel byte for byte; the operand
    copy is its kernel with one comment line more.
    """

    options = CompileOptions.from_environment().with_regblock_enabled(regblock_enabled)
    read = compare_generated_sources(
        permuted_dense_operand_copy(),
        (3, 4),
        [((4, 3), torch.float32)],
        compile_options=options,
    )
    assert "int pA0 = j;" in read.loopir_cpp
    assert "int pA1 = pA0 * A1_size + i;" in read.loopir_cpp
    changed = [
        line
        for line in difflib.ndiff(
            read.legacy_cpp.splitlines(), read.loopir_cpp.splitlines()
        )
        if line[:2] in ("- ", "+ ")
    ]
    assert [(line[:2], line[2:].strip()) for line in changed] == [
        ("+ ", "// Resolve dense coordinates")
    ]

    written = compare_generated_sources(
        permuted_dense_result_copy(),
        (4, 3),
        [((3, 4), torch.float32)],
        compile_options=options,
    )
    assert "int pC0 = j;" in written.loopir_cpp
    assert "int pC1 = pC0 * C1_size + i;" in written.loopir_cpp
    assert written.identical


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["plain", "blocked"])
def test_the_automatic_plan_of_a_permuted_dense_copy(regblock_enabled):
    """The same two copies under the automatic plan, against the shipped compiler.

    The operand copy compiles on both compilers, to different text.  The
    result copy compiles on the new compiler only: the shipped compiler's
    automatic plan stops at ``result_storage_order``, so there is no shipped
    text to compare with.
    """

    options = CompileOptions.from_environment(
        requested_schedule=Schedule()
    ).with_regblock_enabled(regblock_enabled)
    read = compare_generated_sources(
        permuted_dense_operand_copy(),
        (3, 4),
        [((4, 3), torch.float32)],
        compile_options=options,
    )
    assert not read.identical

    kernel = compile_cin_via_loopir(
        permuted_dense_result_copy(),
        (4, 3),
        [((3, 4), torch.float32)],
        compile_options=options,
    )
    assert tuple(kernel.result_mode_order) == (1, 0)
    with pytest.raises(InvalidSchedule) as refusal:
        legacy_generated_cpp(
            permuted_dense_result_copy(),
            (4, 3),
            [((3, 4), torch.float32)],
            compile_options=options,
        )
    assert [d.code for d in refusal.value.diagnostics] == ["result_storage_order"]


@pytest.mark.parametrize("arm", [0, 1], ids=["plain", "blocked"])
@pytest.mark.parametrize(
    "loop_order", [("i", "j"), None], ids=["requested-ij", "automatic"]
)
def test_a_permuted_dense_operand_matches_pytorch(loop_order, arm):
    """``A`` stored ``(j, i)``, executed in float64 on an operand stored that way.

    The requested order ``i, j`` is the nest above: it reads ``A`` against its
    storage order, by coordinate.  The automatic plan is free to pick another.
    """

    from tests.test_scorch.loopir_executed_programs import (
        assert_program_matches_pytorch,
    )

    kernel, _ = assert_program_matches_pytorch(
        (("A", "dd", "ij", (1, 0)),),
        ("ij", "dd", None),
        arm,
        loop_order=loop_order,
    )
    if loop_order is not None:
        assert "int pA1 = pA0 * A1_size + i;" in kernel.cpp_source


@pytest.mark.parametrize("arm", [0, 1], ids=["plain", "blocked"])
def test_a_permuted_dense_result_matches_pytorch(arm):
    """``C`` stored ``(j, i)``, executed in float64 and read back in that order.

    Only the automatic plan runs: a requested order that writes a result
    against its storage order is refused by the plan's ``result_storage_order``
    rule for every result layout, so the nest above is reachable as written
    only, where its kernel is the shipped compiler's byte for byte.
    """

    from tests.test_scorch.loopir_executed_programs import (
        assert_program_matches_pytorch,
    )

    kernel, _ = assert_program_matches_pytorch(
        (("A", "dd", "ij", None),), ("ij", "dd", (1, 0)), arm
    )
    assert tuple(kernel.result_mode_order) == (1, 0)


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
    # int16 is outside the six element types the shipped compiler builds.
    i = IndexVar("i")
    a = TensorVar("a", fmt="d", dtype=torch.int16)
    c = TensorVar("c", fmt="d", dtype=torch.int16)
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
    # The explicit ``+=`` form has no reduction loop either, so it accumulates
    # nothing into a zeroed result and normalizes to the same overwrite store
    # (legacy emits ``=`` for it too); it must not, however, turn the implicit
    # form into an update.
    explicit = lower_normalized_cin_to_loopir(normalize_cin(build(Operation.ADD)))
    assert canonical_program_dump(implicit.program) == canonical_program_dump(
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


def dense_matmul_kij():
    i, k, j = IndexVar("i"), IndexVar("k"), IndexVar("j")
    a = TensorVar("A", fmt="dd")
    b = TensorVar("B", fmt="dd")
    c = TensorVar("C", fmt="dd")
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.MUL, a[i, k], b[k, j]), op=Operation.ADD
    )
    return ForAll(k, ForAll(i, ForAll(j, assign)))


@pytest.mark.parametrize("regblock_enabled", [False, True], ids=["plain", "blocked"])
def test_dense_matmul_lowers_in_kij_order(regblock_enabled):
    """The nest ``k, i, j`` reads ``A[i, k]`` against its storage order.

    That was ``unsupported_loop_order``.  An all-dense operand is read by its
    coordinates in any loop order, so the nest lowers, to the shipped
    compiler's kernel byte for byte.
    """

    lowered = lower_normalized_cin_to_loopir(normalize_cin(dense_matmul_kij()))
    verify_program(lowered.program)
    comparison = compare_generated_sources(
        dense_matmul_kij(),
        (3, 4),
        [((3, 5), torch.float32), ((5, 4), torch.float32)],
        compile_options=CompileOptions.from_environment().with_regblock_enabled(
            regblock_enabled
        ),
    )
    assert "int pA1 = pA0 * A1_size + k;" in comparison.loopir_cpp
    assert comparison.identical


@pytest.mark.parametrize("arm", [0, 1], ids=["plain", "blocked"])
def test_dense_matmul_requested_in_kij_order_matches_pytorch(arm):
    """The same order through the scheduled route, executed in float64."""

    from tests.test_scorch.loopir_executed_programs import (
        assert_program_matches_pytorch,
    )

    kernel, _ = assert_program_matches_pytorch(
        (("A", "dd", "ik", None), ("B", "dd", "kj", None)),
        ("ij", "dd", None),
        arm,
        loop_order=("k", "i", "j"),
    )
    assert "int pA1 = pA0 * A1_size + k;" in kernel.cpp_source


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


@pytest.mark.parametrize("arm", [False, True], ids=["plain", "blocked"])
def test_reduction_into_a_layout_with_a_dense_level_below_a_compressed_one(arm):
    """A reduction whose result coordinate is bound below the reduced loop.

    ``C[i,j,l] = sum_k A[i,j,k,l]`` into ``sds`` was refused until 2026-10-12
    (``unsupported_sparse_output``): the result coordinate ``l`` is bound
    below the reduced loop ``k``, so the sums must be accumulated in a sorted
    coordinate workspace and drained below the dense level ``j``.  It lowers
    now, and is judged on the dense-below-compressed table's inputs.
    """

    from tests.test_scorch.test_loopir_dense_below_compressed_target import (
        assert_statement_matches_pytorch,
    )

    i, j, k, ell = IndexVar("i"), IndexVar("j"), IndexVar("k"), IndexVar("l")
    c = TensorVar("C", fmt="sds", dtype=torch.float32)
    a = TensorVar("A", fmt="ssss", dtype=torch.float32)
    assign = TensorAssign(c[i, j, ell], a[i, j, k, ell], op=Operation.ADD)
    assert_statement_matches_pytorch(
        ForAll(i, ForAll(j, ForAll(k, ForAll(ell, assign)))), arm
    )


@pytest.mark.parametrize(
    "operand_format,result_format", (("ds", "ss"), ("ss", "ds"), ("ds", "sd"))
)
def test_a_sum_into_a_layout_its_rows_do_not_drive_lowers(
    operand_format, result_format
):
    """``C[i,j] = A[i,j] + B[i,j]`` whose result rows are not the operands'.

    These kept a refusal code until the united assembly learned a dense row
    loop above a compressed result row (``ds + ds -> ss``), a union driving a
    dense result row (``ss + ss -> ds``) and a dense block below a compressed
    level (``ds + ds -> sd``).  ``test_loopir_dense_union.py``'s sparse-result
    table executes each on both compilers against float64.
    """

    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt=result_format, dtype=torch.float32)
    a = TensorVar("A", fmt=operand_format, dtype=torch.float32)
    b = TensorVar("B", fmt=operand_format, dtype=torch.float32)
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]))
    lower_normalized_cin_to_loopir(normalize_cin(ForAll(i, ForAll(j, assign))))


def test_copy_into_a_layout_with_a_dense_level_below_a_compressed_one_lowers():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="sds", dtype=torch.float32)
    a = TensorVar("A", fmt="sds", dtype=torch.float32)
    assign = TensorAssign(c[i, j, k], a[i, j, k])
    lower_normalized_cin_to_loopir(
        normalize_cin(ForAll(i, ForAll(j, ForAll(k, assign))))
    )


@pytest.mark.parametrize("arm", [False, True], ids=["plain", "blocked"])
def test_reduction_into_csr_lowers(arm):
    """``C[i,j] = sum_k A[i,j,k]`` into CSR, once refused as
    ``unsupported_sparse_output_reduction``.

    Both result coordinates are bound above the reduced loop, so each stored
    ``(i, j)`` is one scalar accumulation committed after its ``k`` loop.
    ``test_sparse_output_reduction_sweep.py`` also runs this cell (``dss
    ijk->ij as ds``) against the shipped compiler.
    """

    from tests.test_scorch.test_loopir_dense_below_compressed_target import (
        assert_statement_matches_pytorch,
    )

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="ds", dtype=torch.float32)
    a = TensorVar("A", fmt="dss", dtype=torch.float32)
    assign = TensorAssign(c[i, j], a[i, j, k], op=Operation.ADD)
    assert_statement_matches_pytorch(ForAll(i, ForAll(j, ForAll(k, assign))), arm)


@pytest.mark.parametrize("arm", [False, True], ids=["plain", "blocked"])
def test_sum_of_dense_matrices_into_csr(arm):
    """``C[i,j] = A[i,j] + B[i,j]`` with dense operands into CSR.

    Refused as ``unsupported_sparse_output_domain`` until 2026-10-12 because
    the column loop iterates a dense domain.  A dense operand stores every
    cell, so the result stores every coordinate of every row, explicit zeros
    included, and each value is the float32 sum.
    """

    from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="ds", dtype=torch.float32)
    a = TensorVar("A", fmt="dd", dtype=torch.float32)
    b = TensorVar("B", fmt="dd", dtype=torch.float32)
    assign = TensorAssign(c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]))
    for rows, columns in ((3, 4), (0, 3), (2, 0), (1, 1)):
        generator = torch.Generator().manual_seed(rows * 7 + columns)
        left = torch.rand(rows, columns, generator=generator)
        right = torch.rand(rows, columns, generator=generator)
        left[left < 0.4] = 0.0
        right[:, :1] = -left[:, :1]  # the first column sums to an exact zero
        result, _ = execute_cin_via_loopir(
            ForAll(i, ForAll(j, assign)),
            (rows, columns),
            STensor.from_torch(left),
            STensor.from_torch(right),
            compile_options=auto_options(arm, jit=True),
        )
        assert str(result.format) == "d,s"
        positions, coordinates = result.storage.index.mode_indices[1]
        assert positions.tolist() == [columns * row for row in range(rows + 1)]
        assert coordinates.tolist() == list(range(columns)) * rows
        torch.testing.assert_close(
            result.to_torch(in_place=False), left + right, atol=0.0, rtol=0.0
        )


@pytest.mark.parametrize("arm", [False, True], ids=["plain", "blocked"])
def test_dcsr_copy_into_csr(arm):
    """``C[i,j] = A[i,j]`` from DCSR into CSR, once refused as
    ``unsupported_sparse_output_domain`` because the row loop iterates a
    stored row stream.  It lowers now and is judged on the
    dense-below-compressed table's inputs."""

    from tests.test_scorch.test_loopir_dense_below_compressed_target import (
        assert_statement_matches_pytorch,
    )

    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="ds", dtype=torch.float32)
    a = TensorVar("A", fmt="ss", dtype=torch.float32)
    assert_statement_matches_pytorch(
        ForAll(i, ForAll(j, TensorAssign(c[i, j], a[i, j]))), arm
    )


def _united_csr_reduction(reduced, arm):
    """Run ``out[kept] += A[i,j] + B[i,j]`` for CSR ``A`` and ``B`` into a
    dense vector, summing over ``reduced`` (``"j"`` or ``"i"``), and compare
    it with the float64 sum of the dense operands."""

    from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options

    i, j = IndexVar("i"), IndexVar("j")
    kept = i if reduced == "j" else j
    out = TensorVar("y", fmt="d", dtype=torch.float32)
    a = TensorVar("A", fmt="ds", dtype=torch.float32)
    b = TensorVar("B", fmt="ds", dtype=torch.float32)
    assign = TensorAssign(
        out[kept], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]), op=Operation.ADD
    )
    for rows, columns in ((3, 4), (0, 3), (2, 0), (1, 1), (4, 5)):
        generator = torch.Generator().manual_seed(rows * 11 + columns)
        left = torch.rand(rows, columns, generator=generator)
        right = torch.rand(rows, columns, generator=generator)
        left[left < 0.5] = 0.0
        right[right < 0.6] = 0.0
        if rows > 1:
            left[1], right[1] = 0.0, 0.0  # a row neither operand stores
        result, _ = execute_cin_via_loopir(
            ForAll(i, ForAll(j, assign)),
            (rows if reduced == "j" else columns,),
            STensor.from_torch(left.to_sparse_csr()),
            STensor.from_torch(right.to_sparse_csr()),
            compile_options=auto_options(arm, jit=True),
        )
        expected = (left.double() + right.double()).sum(1 if reduced == "j" else 0)
        torch.testing.assert_close(
            result.to_torch(in_place=False).double(), expected, atol=1e-6, rtol=1e-6
        )


@pytest.mark.parametrize("arm", [False, True], ids=["plain", "blocked"])
def test_reduction_over_a_union_into_a_dense_result(arm):
    """``y[i] += A[i,j] + B[i,j]``: the reduced loop is a union.

    Refused as ``unsupported_merged_reduction`` until 2026-10-12.  A dense
    result addresses every cell by its coordinates, so each case of the union
    (both operands store ``j``, or only one does) adds its own term into
    ``y[i]``.
    """

    _united_csr_reduction("j", arm)


@pytest.mark.parametrize("arm", [False, True], ids=["plain", "blocked"])
def test_union_result_coordinate_under_a_reduction(arm):
    """``C[j] += A[i,j] + B[i,j]``: the result coordinate is a union and the
    outer loop is summed over.  Refused as ``unsupported_merged_update`` until
    2026-10-12; the dense result takes ``+=`` in every case of the union.  The
    shape this test held before that, ``C[i, j] += A[i, j] + B[i, j]``, has no
    reduction loop, so its update is plain assignment (see
    ``test_a_merged_add_copy_is_plain_assignment``).
    """

    _united_csr_reduction("i", arm)


def test_a_merged_add_copy_is_plain_assignment():
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    assign = TensorAssign(
        c[i, j], CINBinaryOp(Operation.ADD, a[i, j], b[i, j]), op=Operation.ADD
    )
    lowered = lower_normalized_cin_to_loopir(
        normalize_cin(ForAll(i, ForAll(j, assign)))
    )
    stores = [
        node
        for node in _walk_statements(lowered.program.body)
        if isinstance(node, (Store, StoreReduce))
    ]
    assert len(stores) == 1 and isinstance(stores[0], Store)


def test_analysis_codes_surface_through_the_lowering():
    i, j = IndexVar("i"), IndexVar("j")
    c = TensorVar("C", fmt="ds")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds")
    sub = TensorAssign(c[i, j], CINBinaryOp(Operation.SUB, a[i, j], b[i, j]))
    expect_code("unsupported_sparse_subtraction", ForAll(i, ForAll(j, sub)))

    # A sum with an invariant term stored compressed: the coordinates of j
    # it covers depend on whether x stores i.
    c2 = TensorVar("C2", fmt="dd")
    x = TensorVar("x", fmt="s")
    mixed = TensorAssign(c2[i, j], CINBinaryOp(Operation.ADD, a[i, j], x[i]))
    expect_code("unsupported_union_operand", ForAll(i, ForAll(j, mixed)))


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
