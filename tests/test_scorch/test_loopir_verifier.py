"""Fail-closed coverage for the production LoopIR dense-subset verifier.

Every reachable defect code has at least one direct regression here, built
either through the supported builder API or by forging frozen dataclass state
the way an adversarial caller could.
"""

import dataclasses

import pytest

from scorch.compiler.identity import IndexId, SymbolId
from scorch.compiler.loop_plan import MAX_AFFINE_TILE_WIDTH
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.nodes import (
    BinaryOp,
    Block,
    DenseFor,
    DimensionId,
    Expr,
    IndexValue,
    LevelDecl,
    LevelKind,
    Load,
    LoopIRNodeId,
    LoopProgram,
    ReduceOp,
    ScalarType,
    Stmt,
    StoreReduce,
    TileInnerFor,
    TileOuterFor,
)
from scorch.compiler.loopir.verifier import (
    LoopIRVerificationError,
    verify_program,
)


def expect_defect(code, program):
    with pytest.raises(LoopIRVerificationError) as error:
        verify_program(program)
    assert error.value.defect.code == code, error.value.defect
    return error.value.defect


def forge(node, **fields):
    for name, value in fields.items():
        object.__setattr__(node, name, value)
    return node


@dataclasses.dataclass
class VectorAddFixture:
    builder: LoopIRBuilder
    program: LoopProgram
    dim: DimensionId
    a: SymbolId
    b: SymbolId
    c: SymbolId
    index: IndexId


def build_vector_add(dtype=ScalarType.FLOAT32) -> VectorAddFixture:
    """c[i] = a[i] + b[i] over rank-1 dense tensors."""

    builder = LoopIRBuilder()
    dim = builder.dimension("i")
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    tensors = tuple(
        builder.tensor(symbol, name, dtype, (dim.dimension,), builder.dense_levels(1))
        for symbol, name in ((a, "a"), (b, "b"), (c, "c"))
    )
    index = builder.new_index_id()
    store = builder.store(
        c,
        (builder.index_value(index),),
        builder.binary(
            BinaryOp.ADD,
            builder.load(a, (builder.index_value(index),)),
            builder.load(b, (builder.index_value(index),)),
        ),
    )
    program = builder.program(
        (dim,),
        tensors,
        (a, b),
        (c,),
        builder.block(
            (builder.dense_for(index, dim.dimension, builder.block((store,))),)
        ),
    )
    return VectorAddFixture(builder, program, dim.dimension, a, b, c, index)


@dataclasses.dataclass
class MatmulFixture:
    builder: LoopIRBuilder
    program: LoopProgram
    a: SymbolId
    b: SymbolId
    c: SymbolId


def build_matmul() -> MatmulFixture:
    """C[i, j] += A[i, k] * B[k, j] with loop order (i, k, j)."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_k = builder.dimension("k")
    dim_j = builder.dimension("j")
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    decl_a = builder.tensor(
        a,
        "A",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_k.dimension),
        builder.dense_levels(2),
    )
    decl_b = builder.tensor(
        b,
        "B",
        ScalarType.FLOAT32,
        (dim_k.dimension, dim_j.dimension),
        builder.dense_levels(2),
    )
    decl_c = builder.tensor(
        c,
        "C",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        builder.dense_levels(2),
    )
    index_i = builder.new_index_id()
    index_k = builder.new_index_id()
    index_j = builder.new_index_id()
    leaf = builder.store_reduce(
        c,
        (builder.index_value(index_i), builder.index_value(index_j)),
        ReduceOp.ADD,
        builder.binary(
            BinaryOp.MUL,
            builder.load(
                a, (builder.index_value(index_i), builder.index_value(index_k))
            ),
            builder.load(
                b, (builder.index_value(index_k), builder.index_value(index_j))
            ),
        ),
    )
    body = builder.block((leaf,))
    for index, dim in (
        (index_j, dim_j.dimension),
        (index_k, dim_k.dimension),
        (index_i, dim_i.dimension),
    ):
        body = builder.block((builder.dense_for(index, dim, body),))
    program = builder.program(
        (dim_i, dim_k, dim_j),
        (decl_a, decl_b, decl_c),
        (a, b),
        (c,),
        body,
    )
    return MatmulFixture(builder, program, a, b, c)


def test_valid_programs_verify():
    verify_program(build_vector_add().program)
    verify_program(build_vector_add(ScalarType.FLOAT64).program)
    verify_program(build_matmul().program)


def test_builder_owns_dimension_identity_allocation():
    builder = LoopIRBuilder()
    first = builder.dimension("first")
    second = builder.dimension("second")
    assert first.dimension != second.dimension
    with pytest.raises(TypeError):
        builder.dimension("injected", DimensionId(0))  # type: ignore[call-arg]


def test_non_program_fails_closed():
    expect_defect("malformed_state", object())
    expect_defect("malformed_state", build_vector_add().program.body)


def test_invalid_node_id():
    fixture = build_vector_add()
    forge(fixture.program, node_id="0")
    expect_defect("invalid_node_id", fixture.program)


def test_forged_node_id_value_type():
    fixture = build_vector_add()
    forge(fixture.program, node_id=LoopIRNodeId(True))
    expect_defect("invalid_node_id", fixture.program)


def test_duplicate_node_id():
    fixture = build_vector_add()
    forge(fixture.program.dimensions[0], node_id=fixture.program.node_id)
    expect_defect("duplicate_node_id", fixture.program)


def test_shared_node_object():
    builder = LoopIRBuilder()
    fixture = build_vector_add()
    store = fixture.program.body.statements[0].body.statements[0]
    shared_body = Block(builder._node_id(), (store, store))
    forge(fixture.program.body.statements[0], body=shared_body)
    forge(shared_body, node_id=LoopIRNodeId(9_000))
    expect_defect("shared_node", fixture.program)


def test_cyclic_structure():
    fixture = build_vector_add()
    loop_body = fixture.program.body.statements[0].body
    forge(loop_body, statements=(fixture.program.body,))
    expect_defect("cyclic_structure", fixture.program)


def test_excessive_depth():
    fixture = build_vector_add()
    builder = LoopIRBuilder()
    expr: Expr = builder.load(fixture.a, (builder.index_value(fixture.index),))
    for _ in range(70):
        expr = builder.binary(
            BinaryOp.ADD,
            expr,
            builder.load(fixture.b, (builder.index_value(fixture.index),)),
        )
    store = fixture.program.body.statements[0].body.statements[0]
    forge(store, value=expr)
    # Renumber colliding node ids out of the way of the fixture's own ids.
    seen = [0]

    def renumber(node):
        if isinstance(node, (Expr, Stmt)):
            seen[0] += 1
            object.__setattr__(node, "node_id", LoopIRNodeId(10_000 + seen[0]))
        if type(node) is Load:
            for index in node.indices:
                renumber(index)
        if hasattr(node, "lhs"):
            renumber(node.lhs)
            renumber(node.rhs)

    renumber(expr)
    expect_defect("excessive_depth", fixture.program)


def test_missing_stored_field():
    fixture = build_vector_add()
    del fixture.program.__dict__["tensors"]
    expect_defect("malformed_state", fixture.program)


def test_invalid_symbol_id_in_tensor_decl():
    fixture = build_vector_add()
    forge(fixture.program.tensors[0], symbol="a")
    expect_defect("invalid_symbol_id", fixture.program)


def test_invalid_index_id_on_loop():
    fixture = build_vector_add()
    forge(fixture.program.body.statements[0], index=7)
    expect_defect("invalid_index_id", fixture.program)


def test_invalid_dimension_id_on_loop():
    fixture = build_vector_add()
    forge(fixture.program.body.statements[0], dimension=0)
    expect_defect("invalid_dimension_id", fixture.program)


def test_duplicate_dimension():
    fixture = build_vector_add()
    forge(fixture.program.dimensions[0], dimension=fixture.dim)
    duplicate = dataclasses.replace(
        fixture.program.dimensions[0], node_id=LoopIRNodeId(9_100)
    )
    forge(fixture.program, dimensions=(fixture.program.dimensions[0], duplicate))
    expect_defect("duplicate_dimension", fixture.program)


def test_undefined_dimension_in_tensor():
    fixture = build_vector_add()
    forge(fixture.program.tensors[0], dimensions=(DimensionId(404),))
    expect_defect("undefined_dimension", fixture.program)


def test_undefined_dimension_in_loop():
    fixture = build_vector_add()
    forge(fixture.program.body.statements[0], dimension=DimensionId(404))
    expect_defect("undefined_dimension", fixture.program)


def test_unresolved_dimension():
    fixture = build_vector_add()
    orphan = dataclasses.replace(
        fixture.program.dimensions[0],
        node_id=LoopIRNodeId(9_200),
        dimension=DimensionId(505),
    )
    forge(fixture.program, dimensions=(fixture.program.dimensions[0], orphan))
    forge(fixture.program.body.statements[0], dimension=DimensionId(505))
    expect_defect("unresolved_dimension", fixture.program)


def test_duplicate_symbol_redeclared():
    fixture = build_vector_add()
    forge(fixture.program.tensors[1], symbol=fixture.a)
    expect_defect("duplicate_symbol", fixture.program)


def test_duplicate_symbol_across_roles():
    fixture = build_vector_add()
    forge(fixture.program, inputs=(fixture.a, fixture.b, fixture.c))
    expect_defect("duplicate_symbol", fixture.program)


def test_undefined_tensor_in_roles():
    fixture = build_vector_add()
    forge(fixture.program, inputs=(fixture.a, SymbolId(31_337)))
    expect_defect("undefined_tensor", fixture.program)


def test_output_scope_requires_an_output():
    fixture = build_vector_add()
    forge(fixture.program, inputs=(fixture.a, fixture.b, fixture.c), outputs=())
    expect_defect("output_scope", fixture.program)


def test_output_scope_requires_role_for_every_tensor():
    fixture = build_vector_add()
    forge(fixture.program, inputs=(fixture.a,))
    expect_defect("output_scope", fixture.program)


def test_invalid_scalar_type():
    fixture = build_vector_add()
    forge(fixture.program.tensors[0], dtype="float32")
    expect_defect("invalid_scalar_type", fixture.program)


def test_mixed_dtype():
    fixture = build_vector_add()
    forge(fixture.program.tensors[1], dtype=ScalarType.FLOAT64)
    expect_defect("mixed_dtype", fixture.program)


def test_invalid_mode_order_out_of_range():
    fixture = build_vector_add()
    level = fixture.program.tensors[0].levels[0]
    forge(level, mode=3)
    expect_defect("invalid_mode_order", fixture.program)


def test_invalid_mode_order_not_a_permutation():
    fixture = build_matmul()
    decl = fixture.program.tensors[0]
    forge(decl.levels[1], mode=0)
    expect_defect("invalid_mode_order", fixture.program)


def test_permuted_mode_order_is_structurally_valid():
    """Physically permuted dense layouts verify; the target boundary gates."""

    fixture = build_matmul()
    decl = fixture.program.tensors[0]
    forge(decl.levels[0], mode=1)
    forge(decl.levels[1], mode=0)
    verify_program(fixture.program)


def test_unsupported_level_kind():
    fixture = build_vector_add()
    forge(fixture.program.tensors[0].levels[0], kind=LevelKind.COORDINATE)
    defect = expect_defect("unsupported_level_kind", fixture.program)
    assert "coordinate" in defect.message


def test_compressed_input_level_is_now_declared_executable():
    """Phase 5 deliberately opened COMPRESSED input levels.

    A compressed level no longer fails with ``unsupported_level_kind``; the
    forged program instead fails at the coordinate load, which is only
    defined on all-dense tensors.
    """

    fixture = build_vector_add()
    forge(fixture.program.tensors[0].levels[0], kind=LevelKind.COMPRESSED)
    expect_defect("layout_mismatch", fixture.program)


@pytest.mark.parametrize("kind", [LevelKind.COORDINATE, LevelKind.SINGLETON])
def test_every_unrepresented_level_kind_fails_closed(kind):
    fixture = build_vector_add()
    forge(fixture.program.tensors[2].levels[0], kind=kind)
    expect_defect("unsupported_level_kind", fixture.program)


def test_rank_mismatch_levels_versus_dimensions():
    fixture = build_vector_add()
    builder = LoopIRBuilder()
    extra = LevelDecl(LoopIRNodeId(9_300), LevelKind.DENSE, 1)
    decl = fixture.program.tensors[0]
    forge(decl, levels=(decl.levels[0], extra))
    expect_defect("rank_mismatch", fixture.program)
    del builder


def test_rank_mismatch_on_load():
    fixture = build_matmul()
    leaf = (
        fixture.program.body.statements[0]
        .body.statements[0]
        .body.statements[0]
        .body.statements[0]
    )
    load = leaf.value.lhs
    forge(load, indices=(load.indices[0],))
    expect_defect("rank_mismatch", fixture.program)


def test_unbound_index():
    fixture = build_vector_add()
    store = fixture.program.body.statements[0].body.statements[0]
    forge(store.indices[0], index=IndexId(31_337))
    expect_defect("unbound_index", fixture.program)


def test_duplicate_index_binding():
    fixture = build_vector_add()
    builder = LoopIRBuilder()
    inner = DenseFor(
        LoopIRNodeId(9_400),
        fixture.index,
        fixture.dim,
        fixture.program.body.statements[0].body,
    )
    outer = fixture.program.body.statements[0]
    forge(outer, body=Block(LoopIRNodeId(9_401), (inner,)))
    expect_defect("duplicate_index_binding", fixture.program)
    del builder


def test_domain_mismatch_on_load():
    fixture = build_matmul()
    leaf = (
        fixture.program.body.statements[0]
        .body.statements[0]
        .body.statements[0]
        .body.statements[0]
    )
    load_a = leaf.value.lhs
    # Swap A's [i, k] coordinates so the k coordinate lands in i's domain.
    forge(load_a, indices=(load_a.indices[1], load_a.indices[0]))
    expect_defect("domain_mismatch", fixture.program)


def test_domain_mismatch_on_store():
    fixture = build_matmul()
    leaf = (
        fixture.program.body.statements[0]
        .body.statements[0]
        .body.statements[0]
        .body.statements[0]
    )
    forge(leaf, indices=(leaf.indices[1], leaf.indices[0]))
    expect_defect("domain_mismatch", fixture.program)


def test_type_mismatch_value_position():
    fixture = build_vector_add()
    builder = LoopIRBuilder()
    store = fixture.program.body.statements[0].body.statements[0]
    forge(store, value=IndexValue(LoopIRNodeId(9_500), fixture.index))
    expect_defect("type_mismatch", fixture.program)
    del builder


def test_type_mismatch_coordinate_position():
    fixture = build_vector_add()
    store = fixture.program.body.statements[0].body.statements[0]
    load = store.value.lhs
    forge(store, indices=(load,))
    forge(load, indices=(IndexValue(LoopIRNodeId(9_600), fixture.index),))
    expect_defect("type_mismatch", fixture.program)


def test_unknown_expr():
    class RogueExpr(Expr):
        pass

    fixture = build_vector_add()
    store = fixture.program.body.statements[0].body.statements[0]
    forge(store, value=RogueExpr(LoopIRNodeId(9_700)))
    expect_defect("unknown_expr", fixture.program)


def test_non_expr_fails_closed():
    fixture = build_vector_add()
    store = fixture.program.body.statements[0].body.statements[0]
    forge(store, value="a + b")
    expect_defect("unknown_expr", fixture.program)


def test_unknown_stmt():
    class RogueStmt(Stmt):
        pass

    fixture = build_vector_add()
    loop_body = fixture.program.body.statements[0].body
    forge(loop_body, statements=(RogueStmt(LoopIRNodeId(9_800)),))
    expect_defect("unknown_stmt", fixture.program)


def test_non_stmt_fails_closed():
    fixture = build_vector_add()
    loop_body = fixture.program.body.statements[0].body
    forge(loop_body, statements=("store",))
    expect_defect("unknown_stmt", fixture.program)


def test_output_read():
    fixture = build_vector_add()
    store = fixture.program.body.statements[0].body.statements[0]
    forge(store.value.lhs, tensor=fixture.c)
    expect_defect("output_read", fixture.program)


def test_output_scope_on_store_to_input():
    fixture = build_vector_add()
    store = fixture.program.body.statements[0].body.statements[0]
    forge(store, tensor=fixture.a)
    expect_defect("output_scope", fixture.program)


def test_unwritten_output():
    fixture = build_vector_add()
    loop = fixture.program.body.statements[0]
    forge(loop, body=Block(LoopIRNodeId(9_900), ()))
    expect_defect("unwritten_output", fixture.program)


def test_store_reduce_requires_reduce_op_member():
    fixture = build_matmul()
    leaf = (
        fixture.program.body.statements[0]
        .body.statements[0]
        .body.statements[0]
        .body.statements[0]
    )
    assert type(leaf) is StoreReduce
    forge(leaf, op=BinaryOp.ADD)
    expect_defect("malformed_state", fixture.program)


def test_store_reduce_add_is_the_whole_reduce_surface():
    assert [member.name for member in ReduceOp] == ["ADD"]


def test_malformed_tuple_children():
    fixture = build_vector_add()
    forge(fixture.program, inputs=[fixture.a, fixture.b])
    expect_defect("malformed_state", fixture.program)


def test_malformed_block_statements():
    fixture = build_vector_add()
    loop = fixture.program.body.statements[0]
    forge(loop.body, statements=[loop.body.statements[0]])
    expect_defect("malformed_state", fixture.program)


def test_malformed_tensor_decl_name():
    fixture = build_vector_add()
    forge(fixture.program.tensors[0], name="")
    expect_defect("malformed_state", fixture.program)


def test_malformed_dimension_decl_name():
    fixture = build_vector_add()
    forge(fixture.program.dimensions[0], name=b"i")
    expect_defect("malformed_state", fixture.program)


def test_forged_level_mode_type():
    fixture = build_vector_add()
    forge(fixture.program.tensors[0].levels[0], mode=True)
    expect_defect("malformed_state", fixture.program)


def test_forged_binary_op():
    fixture = build_vector_add()
    store = fixture.program.body.statements[0].body.statements[0]
    forge(store.value, op="+")
    expect_defect("malformed_state", fixture.program)


def test_verifier_reports_paths():
    fixture = build_vector_add()
    forge(fixture.program.tensors[1], dtype=ScalarType.FLOAT64)
    defect = expect_defect("mixed_dtype", fixture.program)
    assert defect.path == "program.tensors[1]"


PRODUCTION_SUBSET_DEFECT_CODES = {
    # Dense-subset codes frozen by Phase 4.
    "malformed_state",
    "invalid_node_id",
    "duplicate_node_id",
    "shared_node",
    "cyclic_structure",
    "excessive_depth",
    "invalid_symbol_id",
    "invalid_index_id",
    "invalid_dimension_id",
    "duplicate_dimension",
    "undefined_dimension",
    "unresolved_dimension",
    "duplicate_symbol",
    "undefined_tensor",
    "invalid_mode_order",
    "invalid_scalar_type",
    "mixed_dtype",
    "rank_mismatch",
    "unsupported_level_kind",
    "unbound_index",
    "duplicate_index_binding",
    "domain_mismatch",
    "type_mismatch",
    "unknown_expr",
    "unknown_stmt",
    "output_read",
    "output_scope",
    "unwritten_output",
    # Sparse-subset codes added by Phase 5.
    "invalid_cursor_id",
    "invalid_position_id",
    "duplicate_cursor_id",
    "duplicate_position_binding",
    "unbound_cursor",
    "unbound_position",
    "parent_position_mismatch",
    "layout_mismatch",
    "merge_domain_mismatch",
    "degenerate_merge",
    "unsupported_sparse_hierarchy",
    "missing_union_default",
    "dead_default",
    "default_contains_cursor",
    "non_leaf_value",
    "unsupported_sparse_output",
    # Affine-split codes added by Phase 6.
    "invalid_tile_id",
    "duplicate_tile_id",
    "unbound_tile",
    "missing_tile_inner",
    "tile_binding_mismatch",
    "invalid_tile_width",
    "tile_index_conflict",
}


def test_defect_codes_are_the_documented_production_subset():
    """Lock the stable defect-code surface of the production verifier."""

    import re

    import scorch.compiler.loopir.verifier as verifier_module

    source = open(verifier_module.__file__).read()
    found = set(re.findall(r"_fail\(\s*\"([a-z_]+)\"", source))
    assert found == PRODUCTION_SUBSET_DEFECT_CODES


# -- Phase-5 sparse subset ----------------------------------------------------


@dataclasses.dataclass
class CsrSpmvFixture:
    builder: LoopIRBuilder
    program: LoopProgram
    a: SymbolId
    x: SymbolId
    y: SymbolId
    row: IndexId
    col: IndexId


def build_csr_spmv() -> CsrSpmvFixture:
    """y[i] += A[i, j] * x[j] with A stored as canonical CSR."""

    from scorch.compiler.loopir.nodes import ReduceOp as _ReduceOp

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, x, y = (builder.new_symbol_id() for _ in range(3))
    decl_a = builder.tensor(
        a,
        "A",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.DENSE, 0),
            builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    decl_x = builder.tensor(
        x, "x", ScalarType.FLOAT32, (dim_j.dimension,), builder.dense_levels(1)
    )
    decl_y = builder.tensor(
        y, "y", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    cursor = builder.new_cursor_id()
    position = builder.new_position_id()
    cursor_decl = builder.sparse_cursor(
        cursor,
        a,
        1,
        builder.dense_position(a, 0, builder.root_position(), builder.index_value(row)),
    )
    leaf = builder.store_reduce(
        y,
        (builder.index_value(row),),
        _ReduceOp.ADD,
        builder.binary(
            BinaryOp.MUL,
            builder.cursor_value(cursor),
            builder.load(x, (builder.index_value(col),)),
        ),
    )
    sparse_loop = builder.sparse_for(cursor_decl, position, col, builder.block((leaf,)))
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_x, decl_y),
        (a, x),
        (y,),
        builder.block(
            (builder.dense_for(row, dim_i.dimension, builder.block((sparse_loop,))),)
        ),
    )
    return CsrSpmvFixture(builder, program, a, x, y, row, col)


@dataclasses.dataclass
class UnionAddFixture:
    builder: LoopIRBuilder
    program: LoopProgram
    a: SymbolId
    b: SymbolId
    c: SymbolId
    row: IndexId
    col: IndexId
    merged: object


def build_union_add(mode=None, with_defaults=True, op=None) -> UnionAddFixture:
    """C[i, j] = A[i, j] + B[i, j] over two CSR inputs into a CSR output."""

    from scorch.compiler.loopir.nodes import MergeMode

    if mode is None:
        mode = MergeMode.UNION
    if op is None:
        op = BinaryOp.ADD if mode is MergeMode.UNION else BinaryOp.MUL
    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    csr_levels = lambda: (  # noqa: E731
        builder.level(LevelKind.DENSE, 0),
        builder.level(LevelKind.COMPRESSED, 1),
    )
    decl_a = builder.tensor(
        a, "A", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), csr_levels()
    )
    decl_b = builder.tensor(
        b, "B", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), csr_levels()
    )
    decl_c = builder.tensor(
        c, "C", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), csr_levels()
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    cursor_a = builder.new_cursor_id()
    cursor_b = builder.new_cursor_id()

    def parent(symbol):
        return builder.dense_position(
            symbol, 0, builder.root_position(), builder.index_value(row)
        )

    decl_cursor_a = builder.sparse_cursor(cursor_a, a, 1, parent(a))
    decl_cursor_b = builder.sparse_cursor(cursor_b, b, 1, parent(b))
    default_a = builder.float_const(0.0) if with_defaults else None
    default_b = builder.float_const(0.0) if with_defaults else None
    leaf = builder.append_entry(
        c,
        (builder.index_value(row), builder.index_value(col)),
        builder.binary(
            op,
            builder.cursor_value(cursor_a, default_a),
            builder.cursor_value(cursor_b, default_b),
        ),
    )
    merged = builder.merged_sparse_for(
        mode, (decl_cursor_a, decl_cursor_b), col, builder.block((leaf,))
    )
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_b, decl_c),
        (a, b),
        (c,),
        builder.block(
            (builder.dense_for(row, dim_i.dimension, builder.block((merged,))),)
        ),
    )
    return UnionAddFixture(builder, program, a, b, c, row, col, merged)


def test_csr_spmv_program_verifies():
    verify_program(build_csr_spmv().program)


def test_union_add_program_verifies():
    verify_program(build_union_add().program)


def test_intersection_multiply_program_verifies():
    from scorch.compiler.loopir.nodes import MergeMode

    fixture = build_union_add(mode=MergeMode.INTERSECTION, with_defaults=False)
    verify_program(fixture.program)


def test_dcsr_single_cursor_descent_verifies():
    """Compressed-under-compressed descent through bound parent positions."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a = builder.new_symbol_id()
    y = builder.new_symbol_id()
    decl_a = builder.tensor(
        a,
        "A",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.COMPRESSED, 0),
            builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    decl_y = builder.tensor(
        y, "y", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    cursor_rows = builder.new_cursor_id()
    cursor_cols = builder.new_cursor_id()
    position_rows = builder.new_position_id()
    position_cols = builder.new_position_id()
    from scorch.compiler.loopir.nodes import ReduceOp as _ReduceOp

    inner = builder.sparse_for(
        builder.sparse_cursor(cursor_cols, a, 1, builder.position_value(position_rows)),
        position_cols,
        col,
        builder.block(
            (
                builder.store_reduce(
                    y,
                    (builder.index_value(row),),
                    _ReduceOp.ADD,
                    builder.cursor_value(cursor_cols),
                ),
            )
        ),
    )
    outer = builder.sparse_for(
        builder.sparse_cursor(cursor_rows, a, 0, builder.root_position()),
        position_rows,
        row,
        builder.block((inner,)),
    )
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_y),
        (a,),
        (y,),
        builder.block((outer,)),
    )
    verify_program(program)


def test_invalid_cursor_id():
    fixture = build_csr_spmv()
    sparse_loop = fixture.program.body.statements[0].body.statements[0]
    forge(sparse_loop.cursor, cursor=41)
    expect_defect("invalid_cursor_id", fixture.program)


def test_invalid_position_id():
    fixture = build_csr_spmv()
    sparse_loop = fixture.program.body.statements[0].body.statements[0]
    forge(sparse_loop, position=17)
    expect_defect("invalid_position_id", fixture.program)


def test_duplicate_cursor_id():
    fixture = build_union_add()
    merged = fixture.program.body.statements[0].body.statements[0]
    forge(merged.cursors[1], cursor=merged.cursors[0].cursor)
    expect_defect("duplicate_cursor_id", fixture.program)


def test_duplicate_position_binding():
    fixture = build_csr_spmv()
    builder = fixture.builder
    outer = fixture.program.body.statements[0]
    sparse_loop = outer.body.statements[0]
    inner_cursor = builder.sparse_cursor(
        builder.new_cursor_id(),
        fixture.a,
        1,
        builder.dense_position(
            fixture.a,
            0,
            builder.root_position(),
            builder.index_value(fixture.row),
        ),
    )
    nested = builder.sparse_for(
        inner_cursor,
        sparse_loop.position,
        builder.new_index_id(),
        sparse_loop.body,
    )
    forge(sparse_loop, body=builder.block((nested,)))
    expect_defect("duplicate_position_binding", fixture.program)


def test_unbound_cursor():
    from scorch.compiler.loopir.nodes import CursorId

    fixture = build_csr_spmv()
    leaf = fixture.program.body.statements[0].body.statements[0].body.statements[0]
    forge(leaf.value.lhs, cursor=CursorId(31_337))
    expect_defect("unbound_cursor", fixture.program)


def test_unbound_position():
    fixture = build_csr_spmv()
    builder = fixture.builder
    sparse_loop = fixture.program.body.statements[0].body.statements[0]
    forge(
        sparse_loop.cursor,
        parent=builder.position_value(builder.new_position_id()),
    )
    expect_defect("unbound_position", fixture.program)


def test_parent_position_mismatch_root_for_level_one():
    fixture = build_csr_spmv()
    builder = fixture.builder
    sparse_loop = fixture.program.body.statements[0].body.statements[0]
    forge(sparse_loop.cursor, parent=builder.root_position())
    expect_defect("parent_position_mismatch", fixture.program)


def test_parent_position_mismatch_wrong_tensor():
    fixture = build_union_add()
    merged = fixture.program.body.statements[0].body.statements[0]
    builder = fixture.builder
    forge(
        merged.cursors[0],
        parent=builder.dense_position(
            fixture.b,
            0,
            builder.root_position(),
            builder.index_value(fixture.row),
        ),
    )
    expect_defect("parent_position_mismatch", fixture.program)


def test_parent_position_mismatch_coordinate_parent():
    fixture = build_csr_spmv()
    builder = fixture.builder
    sparse_loop = fixture.program.body.statements[0].body.statements[0]
    forge(sparse_loop.cursor, parent=builder.index_value(fixture.row))
    expect_defect("parent_position_mismatch", fixture.program)


def test_layout_mismatch_coordinate_load_of_sparse_tensor():
    fixture = build_csr_spmv()
    leaf = fixture.program.body.statements[0].body.statements[0].body.statements[0]
    builder = fixture.builder
    forge(
        leaf.value,
        lhs=builder.load(
            fixture.a,
            (builder.index_value(fixture.row), builder.index_value(fixture.col)),
        ),
    )
    expect_defect("layout_mismatch", fixture.program)


def test_layout_mismatch_cursor_over_dense_level():
    fixture = build_csr_spmv()
    sparse_loop = fixture.program.body.statements[0].body.statements[0]
    forge(sparse_loop.cursor, level=0)
    expect_defect("layout_mismatch", fixture.program)


def test_layout_mismatch_dense_position_on_compressed_level():
    fixture = build_csr_spmv()
    builder = fixture.builder
    sparse_loop = fixture.program.body.statements[0].body.statements[0]
    forge(
        sparse_loop.cursor,
        parent=builder.dense_position(
            fixture.a,
            1,
            builder.dense_position(
                fixture.a,
                0,
                builder.root_position(),
                builder.index_value(fixture.row),
            ),
            builder.index_value(fixture.row),
        ),
    )
    expect_defect("layout_mismatch", fixture.program)


def test_layout_mismatch_store_into_compressed_output():
    fixture = build_union_add()
    builder = fixture.builder
    merged = fixture.program.body.statements[0].body.statements[0]
    leaf = merged.body.statements[0]
    store = builder.store(fixture.c, leaf.coords, leaf.value)
    forge(merged, body=builder.block((store,)))
    expect_defect("layout_mismatch", fixture.program)


def test_layout_mismatch_append_into_dense_output():
    fixture = build_csr_spmv()
    builder = fixture.builder
    sparse_loop = fixture.program.body.statements[0].body.statements[0]
    leaf = sparse_loop.body.statements[0]
    append = builder.append_entry(
        fixture.y, (builder.index_value(fixture.row),), leaf.value
    )
    forge(sparse_loop, body=builder.block((append,)))
    expect_defect("layout_mismatch", fixture.program)


def test_merge_domain_mismatch():
    fixture = build_union_add()
    decl_b = fixture.program.tensors[1]
    dim_i = fixture.program.dimensions[0].dimension
    # B's parent chain stays domain-correct (level 0 still stores dim_i), but
    # its merged leaf level now iterates dim_i beside A's dim_j.
    forge(decl_b, dimensions=(dim_i, dim_i))
    expect_defect("merge_domain_mismatch", fixture.program)


def test_degenerate_merge():
    fixture = build_union_add()
    merged = fixture.program.body.statements[0].body.statements[0]
    forge(merged, cursors=(merged.cursors[0],))
    expect_defect("degenerate_merge", fixture.program)


def test_unsupported_sparse_hierarchy():
    """Merging non-leaf cursors (hierarchical descent) fails closed."""

    fixture = build_union_add()
    decl_a = fixture.program.tensors[0]
    decl_b = fixture.program.tensors[1]
    for decl in (decl_a, decl_b):
        levels = (
            forge(decl.levels[0], kind=LevelKind.COMPRESSED),
            forge(decl.levels[1], kind=LevelKind.COMPRESSED),
        )
        forge(decl, levels=levels)
    merged = fixture.program.body.statements[0].body.statements[0]
    builder = fixture.builder
    forge(merged.cursors[0], level=0, parent=builder.root_position())
    forge(merged.cursors[1], level=0, parent=builder.root_position())
    expect_defect("unsupported_sparse_hierarchy", fixture.program)


def test_missing_union_default():
    fixture = build_union_add(with_defaults=False)
    expect_defect("missing_union_default", fixture.program)


def test_dead_default_in_sparse_for():
    fixture = build_csr_spmv()
    builder = fixture.builder
    leaf = fixture.program.body.statements[0].body.statements[0].body.statements[0]
    forge(leaf.value.lhs, default=builder.float_const(0.0))
    expect_defect("dead_default", fixture.program)


def test_dead_default_in_intersection():
    from scorch.compiler.loopir.nodes import MergeMode

    fixture = build_union_add(mode=MergeMode.INTERSECTION, with_defaults=True)
    expect_defect("dead_default", fixture.program)


def test_default_contains_cursor():
    fixture = build_union_add()
    builder = fixture.builder
    merged = fixture.program.body.statements[0].body.statements[0]
    leaf = merged.body.statements[0]
    forge(
        leaf.value.lhs,
        default=builder.cursor_value(
            merged.cursors[1].cursor, builder.float_const(0.0)
        ),
    )
    expect_defect("default_contains_cursor", fixture.program)


def test_non_leaf_value():
    """Reading a structural (non-leaf) cursor's scalar fails closed."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a = builder.new_symbol_id()
    y = builder.new_symbol_id()
    decl_a = builder.tensor(
        a,
        "A",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.COMPRESSED, 0),
            builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    decl_y = builder.tensor(
        y, "y", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    row = builder.new_index_id()
    cursor_rows = builder.new_cursor_id()
    position_rows = builder.new_position_id()
    from scorch.compiler.loopir.nodes import ReduceOp as _ReduceOp

    outer = builder.sparse_for(
        builder.sparse_cursor(cursor_rows, a, 0, builder.root_position()),
        position_rows,
        row,
        builder.block(
            (
                builder.store_reduce(
                    y,
                    (builder.index_value(row),),
                    _ReduceOp.ADD,
                    builder.cursor_value(cursor_rows),
                ),
            )
        ),
    )
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_y),
        (a,),
        (y,),
        builder.block((outer,)),
    )
    expect_defect("non_leaf_value", program)


def test_unsupported_sparse_output_non_csr():
    fixture = build_union_add()
    decl_c = fixture.program.tensors[2]
    levels = (
        forge(decl_c.levels[0], kind=LevelKind.COMPRESSED, mode=1),
        forge(decl_c.levels[1], kind=LevelKind.DENSE, mode=0),
    )
    forge(decl_c, levels=levels)
    expect_defect("unsupported_sparse_output", fixture.program)


def test_unsupported_sparse_output_rank_three():
    fixture = build_union_add()
    builder = fixture.builder
    dim_extra = builder.dimension("k")
    decl_c = fixture.program.tensors[2]
    forge(
        decl_c,
        dimensions=(*decl_c.dimensions, dim_extra.dimension),
        levels=(
            *decl_c.levels,
            builder.level(LevelKind.COMPRESSED, 2),
        ),
    )
    programs = fixture.program
    forge(programs, dimensions=(*programs.dimensions, dim_extra))
    expect_defect("unsupported_sparse_output", fixture.program)


def test_float_const_must_be_exact_float():
    fixture = build_union_add()
    merged = fixture.program.body.statements[0].body.statements[0]
    leaf = merged.body.statements[0]
    forge(leaf.value.lhs.default, value=0)
    expect_defect("malformed_state", fixture.program)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_float_const_must_be_finite(value):
    fixture = build_union_add()
    merged = fixture.program.body.statements[0].body.statements[0]
    leaf = merged.body.statements[0]
    forge(leaf.value.lhs.default, value=value)
    expect_defect("malformed_state", fixture.program)


def test_merge_mode_member_is_enforced():
    class ForgedMode:
        value = "union"

    fixture = build_union_add()
    merged = fixture.program.body.statements[0].body.statements[0]
    forge(merged, mode=ForgedMode())
    expect_defect("malformed_state", fixture.program)


def test_sparse_nodes_reject_cross_program_sharing():
    fixture = build_union_add()
    merged = fixture.program.body.statements[0].body.statements[0]
    forge(merged, cursors=(merged.cursors[0], merged.cursors[0]))
    expect_defect("shared_node", fixture.program)


# -- Phase-6 affine-split subset ----------------------------------------------


@dataclasses.dataclass
class TiledMatvecFixture:
    builder: LoopIRBuilder
    program: LoopProgram
    outer: TileOuterFor
    inner: TileInnerFor


def build_tiled_matvec(width=4, unroll=False) -> TiledMatvecFixture:
    """y[i] += A[i, j] * x[j] with the j loop affine-split at ``width``."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, x, y = (builder.new_symbol_id() for _ in range(3))
    decl_a = builder.tensor(
        a,
        "A",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        builder.dense_levels(2),
    )
    decl_x = builder.tensor(
        x, "x", ScalarType.FLOAT32, (dim_j.dimension,), builder.dense_levels(1)
    )
    decl_y = builder.tensor(
        y, "y", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    tile = builder.new_tile_id()
    leaf = builder.store_reduce(
        y,
        (builder.index_value(row),),
        ReduceOp.ADD,
        builder.binary(
            BinaryOp.MUL,
            builder.load(a, (builder.index_value(row), builder.index_value(col))),
            builder.load(x, (builder.index_value(col),)),
        ),
    )
    inner = builder.tile_inner_for(
        tile, col, dim_j.dimension, width, unroll, builder.block((leaf,))
    )
    row_loop = builder.dense_for(row, dim_i.dimension, builder.block((inner,)))
    outer = builder.tile_outer_for(
        tile, col, dim_j.dimension, width, builder.block((row_loop,))
    )
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_x, decl_y),
        (a, x),
        (y,),
        builder.block((outer,)),
    )
    assert type(outer) is TileOuterFor and type(inner) is TileInnerFor
    return TiledMatvecFixture(builder, program, outer, inner)


def test_tiled_program_verifies():
    verify_program(build_tiled_matvec().program)
    verify_program(build_tiled_matvec(width=1, unroll=True).program)


def test_tile_ids_must_be_exact_typed_values():
    fixture = build_tiled_matvec()
    forge(fixture.outer, tile="tile0")
    expect_defect("invalid_tile_id", fixture.program)
    fixture = build_tiled_matvec()
    from scorch.compiler.loopir.nodes import TileId

    forge(fixture.inner, tile=TileId("0"))
    expect_defect("invalid_tile_id", fixture.program)


def test_duplicate_tile_ids_fail_closed():
    fixture = build_tiled_matvec()
    builder = fixture.builder
    # Wrap the existing outer loop in a second origin loop reusing its TileId
    # but splitting a fresh index over the same dimension.
    other_index = builder.new_index_id()
    duplicate = TileOuterFor(
        LoopIRNodeId(9_000),
        fixture.outer.tile,
        other_index,
        fixture.outer.dimension,
        4,
        builder.block((fixture.program.body.statements[0],)),
    )
    program = builder.program(
        fixture.program.dimensions,
        fixture.program.tensors,
        fixture.program.inputs,
        fixture.program.outputs,
        builder.block((duplicate,)),
    )
    expect_defect("duplicate_tile_id", program)


def test_point_loop_requires_a_dominating_origin_loop():
    fixture = build_tiled_matvec()
    outer = fixture.program.body.statements[0]
    # Splice the origin loop out: its body becomes the program body.
    program = fixture.builder.program(
        fixture.program.dimensions,
        fixture.program.tensors,
        fixture.program.inputs,
        fixture.program.outputs,
        outer.body,
    )
    expect_defect("unbound_tile", program)


def test_tile_origin_requires_its_point_loop():
    fixture = build_tiled_matvec()
    builder = fixture.builder
    row_loop = fixture.outer.body.statements[0]
    constant_update = builder.store_reduce(
        fixture.program.outputs[0],
        (builder.index_value(row_loop.index),),
        ReduceOp.ADD,
        builder.float_const(1.0),
    )
    forge(row_loop, body=builder.block((constant_update,)))

    defect = expect_defect("missing_tile_inner", fixture.program)
    assert defect.path == "program.body.statements[0]"


def test_point_loop_must_agree_with_its_origin_loop():
    fixture = build_tiled_matvec()
    forge(fixture.inner, width=8)
    expect_defect("tile_binding_mismatch", fixture.program)
    fixture = build_tiled_matvec()
    forge(fixture.inner, index=fixture.builder.new_index_id())
    expect_defect("tile_binding_mismatch", fixture.program)
    fixture = build_tiled_matvec()
    forge(fixture.inner, dimension=fixture.program.dimensions[0].dimension)
    expect_defect("tile_binding_mismatch", fixture.program)


def test_tile_widths_must_be_positive_exact_ints():
    fixture = build_tiled_matvec()
    forge(fixture.outer, width=0)
    forge(fixture.inner, width=0)
    expect_defect("invalid_tile_width", fixture.program)
    fixture = build_tiled_matvec()
    forge(fixture.outer, width=True)
    forge(fixture.inner, width=True)
    expect_defect("invalid_tile_width", fixture.program)
    fixture = build_tiled_matvec()
    forge(fixture.outer, width=4.0)
    forge(fixture.inner, width=4.0)
    expect_defect("invalid_tile_width", fixture.program)
    # Width representability is target-specific: semantic LoopIR and the
    # oracle retain arbitrary positive Python-int widths, while the C++
    # lowering owns its narrower constexpr-int boundary.
    verify_program(build_tiled_matvec(width=MAX_AFFINE_TILE_WIDTH + 1).program)


def test_tile_unroll_must_be_a_bool():
    fixture = build_tiled_matvec()
    forge(fixture.inner, unroll=1)
    expect_defect("malformed_state", fixture.program)


def test_split_index_conflicts_fail_closed():
    # An origin loop for an index bound by an enclosing loop.
    fixture = build_tiled_matvec()
    row_loop = fixture.outer.body.statements[0]
    conflicted = TileOuterFor(
        LoopIRNodeId(9_100),
        fixture.builder.new_tile_id(),
        row_loop.index,
        fixture.program.dimensions[0].dimension,
        2,
        fixture.inner.body,
    )
    forge(row_loop, body=fixture.builder.block((conflicted,)))
    expect_defect("tile_index_conflict", fixture.program)

    # A nested origin loop splitting the same index again.
    fixture = build_tiled_matvec()
    row_loop = fixture.outer.body.statements[0]
    nested = TileOuterFor(
        LoopIRNodeId(9_200),
        fixture.builder.new_tile_id(),
        fixture.outer.index,
        fixture.outer.dimension,
        2,
        fixture.builder.block((fixture.inner,)),
    )
    forge(row_loop, body=fixture.builder.block((nested,)))
    expect_defect("tile_index_conflict", fixture.program)


def test_two_point_loops_for_one_split_rebind_the_coordinate():
    fixture = build_tiled_matvec()
    builder = fixture.builder
    row_loop = fixture.outer.body.statements[0]
    second_inner = builder.tile_inner_for(
        fixture.inner.tile,
        fixture.inner.index,
        fixture.inner.dimension,
        fixture.inner.width,
        False,
        builder.block(()),
    )
    forge(
        row_loop.body.statements[0],
        body=builder.block((second_inner,)),
    )
    expect_defect("duplicate_index_binding", fixture.program)


def test_tile_coordinate_is_unbound_outside_the_point_loop():
    fixture = build_tiled_matvec()
    builder = fixture.builder
    stray = builder.store_reduce(
        fixture.program.outputs[0],
        (builder.index_value(fixture.outer.index),),
        ReduceOp.ADD,
        builder.float_const(1.0),
    )
    forge(fixture.outer, body=builder.block((stray,)))
    expect_defect("unbound_index", fixture.program)


def test_tile_dimension_needs_a_runtime_extent_source():
    fixture = build_tiled_matvec()
    orphan = fixture.builder.dimension("orphan")
    forge(fixture.outer, dimension=orphan.dimension)
    forge(fixture.inner, dimension=orphan.dimension)
    program = fixture.builder.program(
        (*fixture.program.dimensions, orphan),
        fixture.program.tensors,
        fixture.program.inputs,
        fixture.program.outputs,
        fixture.program.body,
    )
    expect_defect("unresolved_dimension", program)


def test_tile_nodes_reject_cycles():
    fixture = build_tiled_matvec()
    forge(fixture.inner, body=fixture.program.body)
    expect_defect("cyclic_structure", fixture.program)
