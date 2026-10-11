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
    MergeMode,
    ParallelDiscipline,
    ParallelIntent,
    ParallelPart,
    PositionId,
    ReduceOp,
    RelayoutId,
    RelayoutScope,
    RelayoutStage,
    ResultTileId,
    ResultTileRegion,
    ScalarType,
    SparseWorkspaceInsert,
    StagedRead,
    Stmt,
    StoreReduce,
    TiledReduce,
    TileId,
    TileInnerFor,
    TileOuterFor,
    WorkspaceReduce,
    WorkspaceRegion,
)
from scorch.compiler.loopir.verifier import (
    LoopIRVerificationError,
    MAX_LOOPIR_TILE_WIDTH,
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

    fixture = build_vector_add()
    level = fixture.program.tensors[0].levels[0]
    forge(level, mode=10**5000)
    defect = expect_defect("invalid_mode_order", fixture.program)
    assert "<integer too large to render>" in defect.message


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
    # Sparse-panel codes added by the Phase-6 panel slice.
    "unbound_panel",
    "missing_panel_window",
    "panel_binding_mismatch",
    "panel_bound_mismatch",
    # Workspace-region codes added by the Phase-6 stack-accumulation slice.
    "invalid_workspace_id",
    "duplicate_workspace_id",
    "unbound_workspace",
    "workspace_scope_mismatch",
    "workspace_write_scope",
    "workspace_read_scope",
    "workspace_coord_mismatch",
    "workspace_output_write",
    "workspace_dead_region",
    # Staged-relayout codes added by the Phase-6 relayout slice.
    "invalid_relayout_id",
    "duplicate_relayout_id",
    "unbound_relayout",
    "relayout_scope_mismatch",
    "relayout_operand_mismatch",
    "relayout_read_mismatch",
    "relayout_dead_region",
    # Heap result-tile codes added by the Phase-6 heap slice.
    "invalid_result_tile_id",
    "duplicate_result_tile_id",
    "unbound_result_tile",
    "result_tile_scope_mismatch",
    "result_tile_result_mismatch",
    "result_tile_write_mismatch",
    "result_tile_residual_write",
    "result_tile_dead_region",
    # Abstract parallel-selection codes added by the Phase-6 parallel slice.
    "invalid_parallel_selection",
    "parallel_target_missing",
    "parallel_work_mismatch",
    "parallel_race",
    # Physical position-load code added by the Phase-7 mixed-operand slice.
    "position_load_mismatch",
    # Assembly-strategy legality code added by the Phase-7 assembly slice.
    "unsupported_assembly_strategy",
    # Accumulation-structure legality code added by the Phase-7 accumulator
    # slice: a wholly dense result has no sparse accumulation workspace, so it
    # cannot record which structure holds one.
    "unsupported_accumulator_structure",
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


def build_sparse_workspace_program():
    builder = LoopIRBuilder()
    dimension = builder.dimension("i")
    output = builder.new_symbol_id()
    output_decl = builder.tensor(
        output,
        "C",
        ScalarType.FLOAT32,
        (dimension.dimension,),
        (builder.level(LevelKind.COMPRESSED, 0),),
    )
    outer_index = builder.new_index_id()
    workspace = builder.new_workspace_id()
    workspace_decl = builder.sparse_workspace_decl(
        workspace,
        "wksp",
        ScalarType.FLOAT32,
        (dimension.dimension,),
    )
    insert = builder.sparse_workspace_insert(
        workspace,
        (builder.index_value(outer_index),),
        ReduceOp.ADD,
        builder.float_const(1.0),
    )
    drain_index = builder.new_index_id()
    append = builder.append_entry(
        output,
        (builder.index_value(drain_index),),
        builder.sparse_workspace_value(workspace),
    )
    drain = builder.sparse_workspace_drain_for(
        workspace,
        (drain_index,),
        builder.block((append,)),
    )
    region = builder.sparse_workspace_region(
        workspace_decl,
        builder.block((insert,)),
        builder.block((drain,)),
    )
    program = builder.program(
        (dimension,),
        (output_decl,),
        (),
        (output,),
        builder.block(
            (
                builder.dense_for(
                    outer_index,
                    dimension.dimension,
                    builder.block((region,)),
                ),
            )
        ),
    )
    return builder, program, region, drain


def test_sparse_workspace_program_verifies():
    _, program, _, _ = build_sparse_workspace_program()
    verify_program(program)


def test_sparse_workspace_drain_must_be_direct_and_dynamic_once():
    builder, program, region, drain = build_sparse_workspace_program()
    repeated = builder.dense_for(
        builder.new_index_id(),
        region.workspace.key_dimensions[0],
        builder.block((drain,)),
    )
    forge(region, consumer=builder.block((repeated,)))
    defect = expect_defect("workspace_read_scope", program)
    assert "directly" in defect.message


def test_sparse_workspace_rejects_nested_same_workspace_drain():
    builder, program, region, drain = build_sparse_workspace_program()
    nested = builder.sparse_workspace_drain_for(
        region.workspace.workspace,
        (builder.new_index_id(),),
        drain.body,
    )
    outer = builder.sparse_workspace_drain_for(
        region.workspace.workspace,
        drain.indices,
        builder.block((nested,)),
    )
    forge(region, consumer=builder.block((outer,)))
    defect = expect_defect("workspace_read_scope", program)
    assert "at most once" in defect.message


def test_sparse_workspace_drain_must_consume_merged_value():
    builder, program, region, drain = build_sparse_workspace_program()
    append = drain.body.statements[0]
    replacement = builder.append_entry(
        append.tensor,
        append.coords,
        builder.float_const(0.0),
    )
    replacement_drain = builder.sparse_workspace_drain_for(
        region.workspace.workspace,
        drain.indices,
        builder.block((replacement,)),
    )
    forge(region, consumer=builder.block((replacement_drain,)))
    defect = expect_defect("workspace_dead_region", program)
    assert "consume" in defect.message


def test_sparse_workspace_malformed_consumer_fails_closed():
    _, program, region, _ = build_sparse_workspace_program()
    object.__delattr__(region.consumer, "statements")
    expect_defect("workspace_read_scope", program)


def test_sparse_workspace_role_and_output_scopes_fail_closed():
    builder, program, region, drain = build_sparse_workspace_program()
    insert = rank_k_insert(region)
    forge(region, producer=builder.block((insert, drain)))
    expect_defect("workspace_read_scope", program)

    builder, program, region, drain = build_sparse_workspace_program()
    append = drain.body.statements[0]
    forge(region, producer=builder.block((*region.producer.statements, append)))
    expect_defect("workspace_output_write", program)

    builder, program, region, _ = build_sparse_workspace_program()
    insert = rank_k_insert(region)
    forged_insert = builder.sparse_workspace_insert(
        region.workspace.workspace,
        insert.coords,
        insert.op,
        builder.sparse_workspace_value(region.workspace.workspace),
    )
    forge(region, producer=builder.block((forged_insert,)))
    expect_defect("workspace_read_scope", program)


def test_merge_positions_have_one_canonical_unbound_spelling():
    fixture = build_union_add(
        mode=MergeMode.INTERSECTION,
        with_defaults=False,
    )
    forge(fixture.merged, positions=(None, None))
    defect = expect_defect("malformed_state", fixture.program)
    assert "canonical empty tuple" in defect.message


def test_resuming_builder_scans_position_ids_inside_merge_tuple():
    fixture = build_union_add(
        mode=MergeMode.INTERSECTION,
        with_defaults=False,
    )
    forge(
        fixture.merged,
        positions=(PositionId(40), PositionId(41)),
    )
    verify_program(fixture.program)
    assert LoopIRBuilder.resuming(fixture.program).new_position_id() == PositionId(42)


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


def test_huge_sparse_and_dense_position_levels_fail_closed():
    huge = 10**5000

    fixture = build_csr_spmv()
    sparse_loop = fixture.program.body.statements[0].body.statements[0]
    forge(sparse_loop.cursor, level=huge)
    defect = expect_defect("rank_mismatch", fixture.program)
    assert "<integer too large to render>" in defect.message

    fixture = build_csr_spmv()
    sparse_loop = fixture.program.body.statements[0].body.statements[0]
    forge(sparse_loop.cursor.parent, level=huge)
    defect = expect_defect("rank_mismatch", fixture.program)
    assert "<integer too large to render>" in defect.message


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
    """A rank-3 DENSE/COMPRESSED layout is now admitted at the output gate
    (the generalized ordered-assembly stream is level-general), so the forged
    program fails closed one boundary later at append-rank reconciliation."""

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
    expect_defect("rank_mismatch", fixture.program)


def test_unsupported_sparse_output_permuted_modes():
    """A non-identity storage order on a sparse output stays fail-closed."""

    fixture = build_union_add()
    builder = fixture.builder
    decl_c = fixture.program.tensors[2]
    forge(
        decl_c,
        levels=(
            builder.level(decl_c.levels[0].kind, 1),
            builder.level(decl_c.levels[1].kind, 0),
        ),
    )
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
    # Semantic LoopIR retains a deliberately huge target-neutral range, while
    # the C++ lowering owns its much narrower constexpr-int boundary.
    verify_program(build_tiled_matvec(width=MAX_AFFINE_TILE_WIDTH + 1).program)
    verify_program(build_tiled_matvec(width=MAX_LOOPIR_TILE_WIDTH).program)
    for width in (MAX_LOOPIR_TILE_WIDTH + 1, 10**5000):
        fixture = build_tiled_matvec(width=width)
        expect_defect("invalid_tile_width", fixture.program)


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


# -- Phase-6 workspace-region subset ------------------------------------------


@dataclasses.dataclass
class StackMatmulFixture:
    builder: LoopIRBuilder
    program: LoopProgram
    a: SymbolId
    b: SymbolId
    c: SymbolId
    row: IndexId
    red: IndexId
    col: IndexId
    outer: TileOuterFor
    region: WorkspaceRegion
    producer_inner: TileInnerFor
    consumer_inner: TileInnerFor
    reduce_stmt: WorkspaceReduce
    copy_out: StoreReduce


def build_stack_matmul(width=4, dtype=ScalarType.FLOAT32) -> StackMatmulFixture:
    """C[i, k] += A[i, j] * B[j, k] with the k loop stack-accumulated.

    The exact shape :func:`apply_stack_tile` produces: the origin loop over
    ``k``, the dense row loop, and a workspace region whose producer runs the
    reduction chain ``j -> k_in`` into the workspace and whose consumer
    copies the tile out with a second point loop of the same split.
    """

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    dim_k = builder.dimension("k")
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    decl_a = builder.tensor(
        a, "A", dtype, (dim_i.dimension, dim_j.dimension), builder.dense_levels(2)
    )
    decl_b = builder.tensor(
        b, "B", dtype, (dim_j.dimension, dim_k.dimension), builder.dense_levels(2)
    )
    decl_c = builder.tensor(
        c, "C", dtype, (dim_i.dimension, dim_k.dimension), builder.dense_levels(2)
    )
    row = builder.new_index_id()
    red = builder.new_index_id()
    col = builder.new_index_id()
    tile = builder.new_tile_id()
    workspace = builder.new_workspace_id()
    workspace_decl = builder.workspace_decl(workspace, "wksp", dtype, tile)
    reduce_stmt = builder.workspace_reduce(
        workspace,
        builder.index_value(col),
        ReduceOp.ADD,
        builder.binary(
            BinaryOp.MUL,
            builder.load(a, (builder.index_value(row), builder.index_value(red))),
            builder.load(b, (builder.index_value(red), builder.index_value(col))),
        ),
    )
    producer_inner = builder.tile_inner_for(
        tile, col, dim_k.dimension, width, False, builder.block((reduce_stmt,))
    )
    producer = builder.block(
        (builder.dense_for(red, dim_j.dimension, builder.block((producer_inner,))),)
    )
    copy_out = builder.store_reduce(
        c,
        (builder.index_value(row), builder.index_value(col)),
        ReduceOp.ADD,
        builder.workspace_read(workspace, builder.index_value(col)),
    )
    consumer_inner = builder.tile_inner_for(
        tile, col, dim_k.dimension, width, False, builder.block((copy_out,))
    )
    region = builder.workspace_region(
        workspace_decl, producer, builder.block((consumer_inner,))
    )
    row_loop = builder.dense_for(row, dim_i.dimension, builder.block((region,)))
    outer = builder.tile_outer_for(
        tile, col, dim_k.dimension, width, builder.block((row_loop,))
    )
    program = builder.program(
        (dim_i, dim_j, dim_k),
        (decl_a, decl_b, decl_c),
        (a, b),
        (c,),
        builder.block((outer,)),
    )
    return StackMatmulFixture(
        builder=builder,
        program=program,
        a=a,
        b=b,
        c=c,
        row=row,
        red=red,
        col=col,
        outer=outer,
        region=region,
        producer_inner=producer_inner,
        consumer_inner=consumer_inner,
        reduce_stmt=reduce_stmt,
        copy_out=copy_out,
    )


def test_stack_workspace_region_verifies():
    verify_program(build_stack_matmul().program)


def test_sibling_point_loops_of_one_split_are_the_moved_boundary():
    """The workspace slice deliberately legalizes sibling point loops.

    The producer and consumer of a region each bind the split coordinate
    once, in disjoint scopes; nested rebinding stays rejected (covered by
    ``test_two_point_loops_for_one_split_rebind_the_coordinate``), and a
    non-point binder still may not rebind a point coordinate.
    """

    fixture = build_stack_matmul()
    verify_program(fixture.program)
    builder = fixture.builder
    stray = builder.dense_for(
        fixture.col,
        fixture.outer.dimension,
        builder.block(
            (
                builder.store_reduce(
                    fixture.c,
                    (
                        builder.index_value(fixture.row),
                        builder.index_value(fixture.col),
                    ),
                    ReduceOp.ADD,
                    builder.float_const(1.0),
                ),
            )
        ),
    )
    forge(
        fixture.region,
        consumer=builder.block((fixture.consumer_inner, stray)),
    )
    expect_defect("duplicate_index_binding", fixture.program)


def test_workspace_id_must_be_typed():
    fixture = build_stack_matmul()
    forge(fixture.region.workspace, workspace=7)
    expect_defect("invalid_workspace_id", fixture.program)
    fixture = build_stack_matmul()
    forge(fixture.reduce_stmt, workspace=object())
    expect_defect("invalid_workspace_id", fixture.program)


def test_workspace_ids_are_declared_once():
    fixture = build_stack_matmul()
    builder = fixture.builder
    # A second sibling region reusing the first region's WorkspaceId.
    second_decl = builder.workspace_decl(
        fixture.region.workspace.workspace,
        "wksp2",
        ScalarType.FLOAT32,
        fixture.outer.tile,
    )
    reduce_two = builder.workspace_reduce(
        fixture.region.workspace.workspace,
        builder.index_value(fixture.col),
        ReduceOp.ADD,
        builder.float_const(1.0),
    )
    producer_two = builder.block(
        (
            builder.tile_inner_for(
                fixture.outer.tile,
                fixture.col,
                fixture.outer.dimension,
                fixture.outer.width,
                False,
                builder.block((reduce_two,)),
            ),
        )
    )
    copy_two = builder.store_reduce(
        fixture.c,
        (builder.index_value(fixture.row), builder.index_value(fixture.col)),
        ReduceOp.ADD,
        builder.workspace_read(
            fixture.region.workspace.workspace, builder.index_value(fixture.col)
        ),
    )
    consumer_two = builder.block(
        (
            builder.tile_inner_for(
                fixture.outer.tile,
                fixture.col,
                fixture.outer.dimension,
                fixture.outer.width,
                False,
                builder.block((copy_two,)),
            ),
        )
    )
    second_region = builder.workspace_region(second_decl, producer_two, consumer_two)
    row_loop = fixture.outer.body.statements[0]
    forge(row_loop, body=builder.block((fixture.region, second_region)))
    expect_defect("duplicate_workspace_id", fixture.program)


def test_workspace_access_requires_an_enclosing_region():
    fixture = build_stack_matmul()
    # Replace the region with its bare producer chain: the reduce now has
    # no enclosing region.
    row_loop = fixture.outer.body.statements[0]
    forge(row_loop, body=fixture.region.producer)
    expect_defect("unbound_workspace", fixture.program)


def test_workspace_region_needs_its_origin_loop_in_scope():
    fixture = build_stack_matmul()
    builder = fixture.builder
    foreign = builder.new_tile_id()
    forge(fixture.region.workspace, tile=foreign)
    expect_defect("workspace_scope_mismatch", fixture.program)


def test_workspace_region_must_open_outside_its_point_loops():
    fixture = build_stack_matmul()
    builder = fixture.builder
    # Wrap the region in a point loop of its own split.
    row_loop = fixture.outer.body.statements[0]
    wrapping_point = builder.tile_inner_for(
        fixture.outer.tile,
        fixture.col,
        fixture.outer.dimension,
        fixture.outer.width,
        False,
        builder.block((fixture.region,)),
    )
    forge(row_loop, body=builder.block((wrapping_point,)))
    expect_defect("workspace_scope_mismatch", fixture.program)


def test_workspace_writes_belong_to_the_producer():
    fixture = build_stack_matmul()
    builder = fixture.builder
    stray_reduce = builder.workspace_reduce(
        fixture.region.workspace.workspace,
        builder.index_value(fixture.col),
        ReduceOp.ADD,
        builder.float_const(1.0),
    )
    forge(
        fixture.consumer_inner,
        body=builder.block((fixture.copy_out, stray_reduce)),
    )
    expect_defect("workspace_write_scope", fixture.program)


def test_workspace_reads_belong_to_the_consumer():
    fixture = build_stack_matmul()
    builder = fixture.builder
    forge(
        fixture.reduce_stmt,
        value=builder.workspace_read(
            fixture.region.workspace.workspace, builder.index_value(fixture.col)
        ),
    )
    expect_defect("workspace_read_scope", fixture.program)


def test_workspace_cells_are_addressed_by_the_owning_point_coordinate():
    fixture = build_stack_matmul()
    builder = fixture.builder
    forge(fixture.reduce_stmt, coord=builder.index_value(fixture.row))
    expect_defect("workspace_coord_mismatch", fixture.program)
    fixture = build_stack_matmul()
    forge(fixture.copy_out.value, coord=fixture.builder.float_const(0.0))
    expect_defect("workspace_coord_mismatch", fixture.program)


def test_workspace_producer_must_not_write_outputs():
    fixture = build_stack_matmul()
    builder = fixture.builder
    stray_store = builder.store_reduce(
        fixture.c,
        (builder.index_value(fixture.row), builder.index_value(fixture.col)),
        ReduceOp.ADD,
        builder.float_const(1.0),
    )
    forge(
        fixture.producer_inner,
        body=builder.block((fixture.reduce_stmt, stray_store)),
    )
    expect_defect("workspace_output_write", fixture.program)


def test_workspace_regions_must_accumulate_and_copy_out():
    fixture = build_stack_matmul()
    builder = fixture.builder
    constant_store = builder.store_reduce(
        fixture.c,
        (builder.index_value(fixture.row), builder.index_value(fixture.col)),
        ReduceOp.ADD,
        builder.float_const(1.0),
    )
    forge(fixture.producer_inner, body=builder.block((constant_store,)))
    expect_defect("workspace_output_write", fixture.program)

    fixture = build_stack_matmul()
    builder = fixture.builder
    silent_point = builder.tile_inner_for(
        fixture.outer.tile,
        fixture.col,
        fixture.outer.dimension,
        fixture.outer.width,
        False,
        builder.block(()),
    )
    forge(fixture.region, producer=builder.block((silent_point,)))
    expect_defect("workspace_dead_region", fixture.program)

    fixture = build_stack_matmul()
    builder = fixture.builder
    blind_copy = builder.store_reduce(
        fixture.c,
        (builder.index_value(fixture.row), builder.index_value(fixture.col)),
        ReduceOp.ADD,
        builder.float_const(1.0),
    )
    forge(fixture.consumer_inner, body=builder.block((blind_copy,)))
    expect_defect("workspace_dead_region", fixture.program)


def test_workspace_decl_state_is_typed():
    fixture = build_stack_matmul()
    forge(fixture.region.workspace, name="")
    expect_defect("malformed_state", fixture.program)
    fixture = build_stack_matmul()
    forge(fixture.region.workspace, dtype="float32")
    expect_defect("invalid_scalar_type", fixture.program)
    fixture = build_stack_matmul()
    forge(fixture.region.workspace, dtype=ScalarType.FLOAT64)
    expect_defect("mixed_dtype", fixture.program)
    fixture = build_stack_matmul()
    forge(fixture.region.workspace, tile=17)
    expect_defect("invalid_tile_id", fixture.program)
    fixture = build_stack_matmul()
    forge(fixture.region, workspace=fixture.reduce_stmt)
    expect_defect("malformed_state", fixture.program)


def test_workspace_region_bodies_must_be_blocks():
    fixture = build_stack_matmul()
    forge(fixture.region, producer=fixture.reduce_stmt)
    expect_defect("malformed_state", fixture.program)
    fixture = build_stack_matmul()
    forge(fixture.region, consumer=None)
    expect_defect("malformed_state", fixture.program)


def test_workspace_reduce_state_is_typed():
    fixture = build_stack_matmul()
    forge(fixture.reduce_stmt, op="add")
    expect_defect("malformed_state", fixture.program)
    fixture = build_stack_matmul()
    forge(fixture.reduce_stmt, value=fixture.builder.index_value(fixture.col))
    expect_defect("type_mismatch", fixture.program)


def test_workspace_region_rejects_cycles_and_shared_nodes():
    fixture = build_stack_matmul()
    forge(fixture.region, producer=fixture.program.body)
    expect_defect("cyclic_structure", fixture.program)
    fixture = build_stack_matmul()
    forge(fixture.region, consumer=fixture.region.producer)
    expect_defect("shared_node", fixture.program)


# -- Phase-6 sparse coordinate panels -----------------------------------------


@dataclasses.dataclass
class PanelSpmmFixture:
    builder: LoopIRBuilder
    program: LoopProgram
    a: SymbolId
    b: SymbolId
    c: SymbolId
    dim_i: DimensionId
    dim_j: DimensionId
    dim_k: DimensionId
    row: IndexId
    col: IndexId
    free: IndexId
    panel: object
    window: object
    row_loop: object
    free_loop: object


def build_panel_spmm(width=3, dtype=ScalarType.FLOAT32) -> PanelSpmmFixture:
    """C[i, k] += A[i, j] * B[j, k] with a width-``width`` panel over j.

    A is canonical CSR, B and C are dense; the panel origin loop wraps the
    dense row loop and the window replaces the compressed coordinate loop —
    the legacy SpMM tile-j shape.
    """

    from scorch.compiler.loopir.nodes import ReduceOp as _ReduceOp

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    dim_k = builder.dimension("k")
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    decl_a = builder.tensor(
        a,
        "A",
        dtype,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.DENSE, 0),
            builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    decl_b = builder.tensor(
        b,
        "B",
        dtype,
        (dim_j.dimension, dim_k.dimension),
        builder.dense_levels(2),
    )
    decl_c = builder.tensor(
        c,
        "C",
        dtype,
        (dim_i.dimension, dim_k.dimension),
        builder.dense_levels(2),
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    free = builder.new_index_id()
    cursor = builder.new_cursor_id()
    position = builder.new_position_id()
    tile = builder.new_tile_id()
    cursor_decl = builder.sparse_cursor(
        cursor,
        a,
        1,
        builder.dense_position(a, 0, builder.root_position(), builder.index_value(row)),
    )
    leaf = builder.store_reduce(
        c,
        (builder.index_value(row), builder.index_value(free)),
        _ReduceOp.ADD,
        builder.binary(
            BinaryOp.MUL,
            builder.cursor_value(cursor),
            builder.load(b, (builder.index_value(col), builder.index_value(free))),
        ),
    )
    free_loop = builder.dense_for(free, dim_k.dimension, builder.block((leaf,)))
    window = builder.sparse_window_for(
        tile, cursor_decl, position, col, builder.block((free_loop,))
    )
    row_loop = builder.dense_for(row, dim_i.dimension, builder.block((window,)))
    panel = builder.panel_outer_for(
        tile,
        col,
        dim_j.dimension,
        width,
        b,
        0,
        builder.block((row_loop,)),
    )
    program = builder.program(
        (dim_i, dim_j, dim_k),
        (decl_a, decl_b, decl_c),
        (a, b),
        (c,),
        builder.block((panel,)),
    )
    return PanelSpmmFixture(
        builder,
        program,
        a,
        b,
        c,
        dim_i.dimension,
        dim_j.dimension,
        dim_k.dimension,
        row,
        col,
        free,
        panel,
        window,
        row_loop,
        free_loop,
    )


def test_panel_spmm_fixture_verifies():
    verify_program(build_panel_spmm().program)
    verify_program(build_panel_spmm(width=1).program)
    verify_program(build_panel_spmm(width=10**9).program)


def test_panel_tile_identity_is_typed_and_unique():
    fixture = build_panel_spmm()
    forge(fixture.panel, tile=7)
    expect_defect("invalid_tile_id", fixture.program)

    fixture = build_panel_spmm()
    forge(fixture.window, tile="0")
    expect_defect("invalid_tile_id", fixture.program)

    # A second panel (or an affine split) may not reuse the panel's TileId:
    # panels draw from the same identity space as affine splits.
    fixture = build_panel_spmm()
    duplicate = fixture.builder.panel_outer_for(
        fixture.panel.tile,
        fixture.builder.new_index_id(),
        fixture.dim_j,
        4,
        fixture.b,
        0,
        fixture.builder.block(()),
    )
    forge(
        fixture.program,
        body=fixture.builder.block((fixture.panel, duplicate)),
    )
    defect = expect_defect("duplicate_tile_id", fixture.program)
    assert "reused" in defect.message

    fixture = build_panel_spmm()
    inner = TileInnerFor(
        LoopIRNodeId(10_000),
        fixture.panel.tile,
        fixture.free,
        fixture.dim_k,
        3,
        False,
        fixture.builder.block(()),
    )
    forge(fixture.free_loop, body=fixture.builder.block((inner,)))
    # An affine point loop cannot bind a panel's tile: panels are not open
    # affine splits.
    expect_defect("unbound_tile", fixture.program)


def test_panel_width_must_be_a_positive_exact_int():
    for width in (0, -3, True, 2.0, None):
        fixture = build_panel_spmm()
        forge(fixture.panel, width=width)
        expect_defect("invalid_tile_width", fixture.program)

    verify_program(build_panel_spmm(width=MAX_LOOPIR_TILE_WIDTH).program)
    for width in (MAX_LOOPIR_TILE_WIDTH + 1, 10**5000):
        fixture = build_panel_spmm(width=width)
        expect_defect("invalid_tile_width", fixture.program)

    # Both reader surfaces verify before rendering, so an over-limit semantic
    # integer remains a controlled LoopIR defect rather than leaking
    # CPython's decimal-conversion ValueError.
    from scorch.compiler.loopir.printer import (
        canonical_program_dump,
        print_program,
    )

    for render in (print_program, canonical_program_dump):
        fixture = build_panel_spmm(width=10**5000)
        with pytest.raises(LoopIRVerificationError) as error:
            render(fixture.program)
        assert error.value.defect.code == "invalid_tile_width"


def test_panel_huge_integer_diagnostics_fail_closed():
    huge = 10**5000

    fixture = build_panel_spmm()
    forge(fixture.panel, bound_level=huge)
    defect = expect_defect("rank_mismatch", fixture.program)
    assert "rank-2" in defect.message

    fixture = build_panel_spmm()
    forge(fixture.window, tile=TileId(huge))
    defect = expect_defect("unbound_panel", fixture.program)
    assert "<integer too large to render>" in defect.message

    fixture = build_panel_spmm()
    forge(fixture.panel, tile=TileId(huge))
    plain = fixture.builder.sparse_for(
        fixture.window.cursor,
        fixture.window.position,
        fixture.window.coord_index,
        fixture.window.body,
    )
    forge(fixture.row_loop, body=fixture.builder.block((plain,)))
    defect = expect_defect("missing_panel_window", fixture.program)
    assert "<integer too large to render>" in defect.message


def test_panel_index_conflicts_fail_closed():
    # The panel's logical index is already bound by an enclosing loop.
    fixture = build_panel_spmm()
    wrapper = fixture.builder.dense_for(
        fixture.builder.new_index_id(), fixture.dim_j, fixture.builder.block(())
    )
    forge(wrapper, index=fixture.col, body=fixture.builder.block((fixture.panel,)))
    forge(fixture.program, body=fixture.builder.block((wrapper,)))
    expect_defect("tile_index_conflict", fixture.program)

    # An affine origin loop may not split an index a panel already owns.
    fixture = build_panel_spmm()
    outer = TileOuterFor(
        LoopIRNodeId(10_001),
        fixture.builder.new_tile_id(),
        fixture.col,
        fixture.dim_j,
        2,
        fixture.builder.block((fixture.window,)),
    )
    forge(fixture.row_loop, body=fixture.builder.block((outer,)))
    expect_defect("tile_index_conflict", fixture.program)


def test_panel_bound_must_name_a_declared_dense_level_of_its_dimension():
    fixture = build_panel_spmm()
    forge(fixture.panel, bound_tensor=SymbolId(999_999))
    expect_defect("undefined_tensor", fixture.program)

    fixture = build_panel_spmm()
    forge(fixture.panel, bound_tensor="B")
    expect_defect("invalid_symbol_id", fixture.program)

    fixture = build_panel_spmm()
    forge(fixture.panel, bound_level=2)
    expect_defect("rank_mismatch", fixture.program)

    fixture = build_panel_spmm()
    forge(fixture.panel, bound_level=True)
    expect_defect("malformed_state", fixture.program)

    # A's level 1 stores dimension j but is COMPRESSED: the bound must be a
    # dense extent source.
    fixture = build_panel_spmm()
    forge(fixture.panel, bound_tensor=fixture.a, bound_level=1)
    expect_defect("panel_bound_mismatch", fixture.program)

    # B's level 1 is dense but stores dimension k, not the panel dimension.
    fixture = build_panel_spmm()
    forge(fixture.panel, bound_level=1)
    expect_defect("panel_bound_mismatch", fixture.program)


def test_panel_origin_requires_its_window():
    fixture = build_panel_spmm()
    plain = fixture.builder.sparse_for(
        fixture.window.cursor,
        fixture.window.position,
        fixture.window.coord_index,
        fixture.window.body,
    )
    forge(fixture.row_loop, body=fixture.builder.block((plain,)))
    expect_defect("missing_panel_window", fixture.program)


def test_window_requires_its_dominating_panel():
    # No panel at all: the window's tile is unbound.
    fixture = build_panel_spmm()
    forge(fixture.program, body=fixture.builder.block((fixture.row_loop,)))
    expect_defect("unbound_panel", fixture.program)

    # A window under a different panel's scope is still unbound.
    fixture = build_panel_spmm()
    forge(fixture.window, tile=fixture.builder.new_tile_id())
    expect_defect("unbound_panel", fixture.program)

    # An affine origin loop does not open a panel scope.
    fixture = build_panel_spmm()
    affine_tile = fixture.builder.new_tile_id()
    forge(fixture.window, tile=affine_tile)
    inner_body = fixture.builder.block((fixture.row_loop,))
    outer = TileOuterFor(
        LoopIRNodeId(10_002),
        affine_tile,
        fixture.builder.new_index_id(),
        fixture.dim_j,
        3,
        inner_body,
    )
    forge(fixture.program, body=fixture.builder.block((outer,)))
    expect_defect("unbound_panel", fixture.program)


def test_window_must_agree_with_its_panel():
    # The window must bind the panel's logical index.
    fixture = build_panel_spmm()
    forge(fixture.panel, index=fixture.builder.new_index_id())
    expect_defect("panel_binding_mismatch", fixture.program)

    # The window's cursor level must store the panel's dimension.
    fixture = build_panel_spmm()
    forge(
        fixture.panel,
        dimension=fixture.dim_i,
        bound_tensor=fixture.a,
        bound_level=0,
    )
    expect_defect("panel_binding_mismatch", fixture.program)


def test_window_binds_coordinate_and_position_once_only():
    # A second window of the same panel would rebind the coordinate: the
    # sibling-rebinding boundary stays owned by the workspace point family.
    fixture = build_panel_spmm()
    second_cursor = fixture.builder.sparse_cursor(
        fixture.builder.new_cursor_id(),
        fixture.a,
        1,
        fixture.builder.dense_position(
            fixture.a,
            0,
            fixture.builder.root_position(),
            fixture.builder.index_value(fixture.row),
        ),
    )
    second = fixture.builder.sparse_window_for(
        fixture.panel.tile,
        second_cursor,
        fixture.builder.new_position_id(),
        fixture.col,
        fixture.builder.block(()),
    )
    forge(
        fixture.row_loop,
        body=fixture.builder.block((fixture.window, second)),
    )
    expect_defect("duplicate_index_binding", fixture.program)

    # A nested window reusing the outer window's PositionId is rejected
    # before its (would-be duplicate) coordinate binding.
    fixture = build_panel_spmm()
    duplicate_position = fixture.builder.sparse_window_for(
        fixture.panel.tile,
        fixture.builder.sparse_cursor(
            fixture.builder.new_cursor_id(),
            fixture.a,
            1,
            fixture.builder.dense_position(
                fixture.a,
                0,
                fixture.builder.root_position(),
                fixture.builder.index_value(fixture.row),
            ),
        ),
        fixture.window.position,
        fixture.col,
        fixture.builder.block(()),
    )
    forge(fixture.free_loop, body=fixture.builder.block((duplicate_position,)))
    defect = expect_defect("duplicate_position_binding", fixture.program)
    assert "bound more than once" in defect.message


def test_window_cursor_inherits_the_cursor_discipline():
    fixture = build_panel_spmm()
    forge(fixture.window.cursor, level=0)
    expect_defect("layout_mismatch", fixture.program)

    fixture = build_panel_spmm()
    forge(fixture.window, cursor=fixture.builder.index_value(fixture.col))
    expect_defect("malformed_state", fixture.program)


def test_panel_nodes_reject_hostile_subclasses_and_cycles():
    from scorch.compiler.loopir.nodes import PanelOuterFor, SparseWindowFor

    class HostilePanel(PanelOuterFor):
        pass

    fixture = build_panel_spmm()
    hostile = HostilePanel(
        LoopIRNodeId(10_003),
        fixture.panel.tile,
        fixture.col,
        fixture.dim_j,
        3,
        fixture.b,
        0,
        fixture.builder.block(()),
    )
    forge(fixture.program, body=fixture.builder.block((hostile,)))
    expect_defect("unknown_stmt", fixture.program)

    class HostileWindow(SparseWindowFor):
        pass

    fixture = build_panel_spmm()
    hostile_window = HostileWindow(
        LoopIRNodeId(10_004),
        fixture.window.tile,
        fixture.window.cursor,
        fixture.window.position,
        fixture.window.coord_index,
        fixture.window.body,
    )
    forge(fixture.row_loop, body=fixture.builder.block((hostile_window,)))
    expect_defect("unknown_stmt", fixture.program)

    fixture = build_panel_spmm()
    forge(fixture.panel, body=fixture.program.body)
    expect_defect("cyclic_structure", fixture.program)

    fixture = build_panel_spmm()
    forge(fixture.program, body=fixture.builder.block((fixture.panel, fixture.panel)))
    expect_defect("shared_node", fixture.program)


def test_panel_defect_paths_are_reported():
    fixture = build_panel_spmm()
    forge(fixture.panel, bound_level=1)
    defect = expect_defect("panel_bound_mismatch", fixture.program)
    assert defect.path == "program.body.statements[0].bound_level"


# -- Phase-6 staged-operand relayout regions ----------------------------------


@dataclasses.dataclass
class RelayoutSpmmFixture:
    builder: LoopIRBuilder
    program: LoopProgram
    a: SymbolId
    b: SymbolId
    c: SymbolId
    dim_i: DimensionId
    dim_j: DimensionId
    dim_k: DimensionId
    row: IndexId
    col: IndexId
    free: IndexId
    pack_tile: TileId
    panel_tile: TileId
    relayout: RelayoutId
    decl: object
    stage: object
    staged: object
    pack: object
    panel: object
    window: object
    row_loop: object
    pack_point: object
    leaf: object


def build_relayout_spmm(
    scope=RelayoutScope.PANEL,
    width=3,
    strip=4,
    dtype=ScalarType.FLOAT32,
) -> RelayoutSpmmFixture:
    """C[i, k] += A[i, j] * staged B[j, k] — the packed tile-ijk shape.

    An outermost affine pack tile over ``k``, a sparse panel over ``j``
    directly inside it, and one staging region at the requested scope
    reading B through :class:`StagedRead` in the compute leaf.
    """

    from scorch.compiler.loopir.nodes import ReduceOp as _ReduceOp

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    dim_k = builder.dimension("k")
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    decl_a = builder.tensor(
        a,
        "A",
        dtype,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.DENSE, 0),
            builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    decl_b = builder.tensor(
        b,
        "B",
        dtype,
        (dim_j.dimension, dim_k.dimension),
        builder.dense_levels(2),
    )
    decl_c = builder.tensor(
        c,
        "C",
        dtype,
        (dim_i.dimension, dim_k.dimension),
        builder.dense_levels(2),
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    free = builder.new_index_id()
    cursor = builder.new_cursor_id()
    position = builder.new_position_id()
    pack_tile = builder.new_tile_id()
    panel_tile = builder.new_tile_id()
    relayout = builder.new_relayout_id()
    cursor_decl = builder.sparse_cursor(
        cursor,
        a,
        1,
        builder.dense_position(a, 0, builder.root_position(), builder.index_value(row)),
    )
    staged = builder.staged_read(
        relayout, (builder.index_value(col), builder.index_value(free))
    )
    leaf = builder.store_reduce(
        c,
        (builder.index_value(row), builder.index_value(free)),
        _ReduceOp.ADD,
        builder.binary(BinaryOp.MUL, builder.cursor_value(cursor), staged),
    )
    pack_point = builder.tile_inner_for(
        pack_tile, free, dim_k.dimension, strip, False, builder.block((leaf,))
    )
    window = builder.sparse_window_for(
        panel_tile, cursor_decl, position, col, builder.block((pack_point,))
    )
    row_loop = builder.dense_for(row, dim_i.dimension, builder.block((window,)))
    decl = builder.relayout_decl(relayout, b, panel_tile, pack_tile, scope)
    if scope is RelayoutScope.PANEL:
        stage = builder.relayout_stage(decl, builder.block((row_loop,)))
        panel = builder.panel_outer_for(
            panel_tile, col, dim_j.dimension, width, b, 0, builder.block((stage,))
        )
        pack = builder.tile_outer_for(
            pack_tile, free, dim_k.dimension, strip, builder.block((panel,))
        )
    else:
        panel = builder.panel_outer_for(
            panel_tile, col, dim_j.dimension, width, b, 0, builder.block((row_loop,))
        )
        stage = builder.relayout_stage(decl, builder.block((panel,)))
        pack = builder.tile_outer_for(
            pack_tile, free, dim_k.dimension, strip, builder.block((stage,))
        )
    program = builder.program(
        (dim_i, dim_j, dim_k),
        (decl_a, decl_b, decl_c),
        (a, b),
        (c,),
        builder.block((pack,)),
    )
    return RelayoutSpmmFixture(
        builder,
        program,
        a,
        b,
        c,
        dim_i.dimension,
        dim_j.dimension,
        dim_k.dimension,
        row,
        col,
        free,
        pack_tile,
        panel_tile,
        relayout,
        decl,
        stage,
        staged,
        pack,
        panel,
        window,
        row_loop,
        pack_point,
        leaf,
    )


def test_relayout_fixture_verifies_in_both_scopes():
    verify_program(build_relayout_spmm(RelayoutScope.PANEL).program)
    verify_program(build_relayout_spmm(RelayoutScope.PACK_AXIS).program)
    verify_program(build_relayout_spmm(width=1, strip=1).program)
    verify_program(build_relayout_spmm(dtype=ScalarType.FLOAT64).program)
    verify_program(
        build_relayout_spmm(width=MAX_LOOPIR_TILE_WIDTH, strip=10**9).program
    )


def test_relayout_identity_is_typed_and_unique():
    fixture = build_relayout_spmm()
    forge(fixture.decl, relayout=7)
    expect_defect("invalid_relayout_id", fixture.program)

    fixture = build_relayout_spmm()
    forge(fixture.staged, relayout="0")
    expect_defect("invalid_relayout_id", fixture.program)

    class HostileRelayoutId(RelayoutId):
        pass

    fixture = build_relayout_spmm()
    forge(fixture.decl, relayout=HostileRelayoutId(fixture.relayout.value))
    expect_defect("invalid_relayout_id", fixture.program)

    fixture = build_relayout_spmm()
    forge(fixture.decl, relayout=RelayoutId(True))
    expect_defect("invalid_relayout_id", fixture.program)

    # A second region may not reuse an already-declared relayout identity.
    fixture = build_relayout_spmm()
    inner_decl = fixture.builder.relayout_decl(
        fixture.relayout,
        fixture.b,
        fixture.panel_tile,
        fixture.pack_tile,
        RelayoutScope.PANEL,
    )
    inner_stage = fixture.builder.relayout_stage(inner_decl, fixture.stage.body)
    forge(fixture.stage, body=fixture.builder.block((inner_stage,)))
    expect_defect("duplicate_relayout_id", fixture.program)


def test_staged_read_requires_an_enclosing_region():
    # The staged read sits under the panel but no region is open.
    fixture = build_relayout_spmm()
    forge(fixture.panel, body=fixture.stage.body)
    expect_defect("unbound_relayout", fixture.program)

    # A second read while the region remains open is legal.
    fixture = build_relayout_spmm()
    stray = fixture.builder.staged_read(
        fixture.relayout,
        (
            fixture.builder.index_value(fixture.col),
            fixture.builder.index_value(fixture.free),
        ),
    )
    stray_leaf = fixture.builder.store_reduce(
        fixture.c,
        (
            fixture.builder.index_value(fixture.row),
            fixture.builder.index_value(fixture.free),
        ),
        ReduceOp.ADD,
        stray,
    )
    forge(
        fixture.pack_point,
        body=fixture.builder.block((fixture.leaf, stray_leaf)),
    )
    verify_program(fixture.program)

    # A next sibling after the region exits may not reuse its identity.
    # Move the region down around the original leaf so the point/window
    # binders remain live when the stray sibling is checked.
    fixture = build_relayout_spmm()
    stray = fixture.builder.staged_read(
        fixture.relayout,
        (
            fixture.builder.index_value(fixture.col),
            fixture.builder.index_value(fixture.free),
        ),
    )
    stray_leaf = fixture.builder.store_reduce(
        fixture.c,
        (
            fixture.builder.index_value(fixture.row),
            fixture.builder.index_value(fixture.free),
        ),
        ReduceOp.ADD,
        stray,
    )
    local_stage = fixture.builder.relayout_stage(
        fixture.decl,
        fixture.builder.block((fixture.leaf,)),
    )
    forge(
        fixture.pack_point,
        body=fixture.builder.block((local_stage, stray_leaf)),
    )
    forge(fixture.panel, body=fixture.stage.body)
    expect_defect("unbound_relayout", fixture.program)

    fixture = build_relayout_spmm()
    huge = RelayoutId(10**5000)
    forge(fixture.staged, relayout=huge)
    defect = expect_defect("unbound_relayout", fixture.program)
    assert "too large" in defect.message


def test_relayout_region_scope_discipline():
    # The pack split's origin loop must be open.
    fixture = build_relayout_spmm()
    forge(fixture.decl, pack=fixture.builder.new_tile_id())
    expect_defect("relayout_scope_mismatch", fixture.program)

    # PANEL scope outside the panel origin fails closed.
    fixture = build_relayout_spmm(RelayoutScope.PACK_AXIS)
    forge(fixture.decl, scope=RelayoutScope.PANEL)
    expect_defect("relayout_scope_mismatch", fixture.program)

    # PACK_AXIS scope inside the panel origin fails closed.
    fixture = build_relayout_spmm(RelayoutScope.PANEL)
    forge(fixture.decl, scope=RelayoutScope.PACK_AXIS)
    expect_defect("relayout_scope_mismatch", fixture.program)

    # A panel identity that names an affine split is not a panel scope.
    fixture = build_relayout_spmm(RelayoutScope.PANEL)
    forge(fixture.decl, panel=fixture.pack_tile)
    expect_defect("relayout_scope_mismatch", fixture.program)


def test_relayout_operand_structure_is_verified():
    # Rank-1 operands are outside the family.
    fixture = build_relayout_spmm()
    operand_decl = fixture.program.tensors[1]
    forge(
        operand_decl,
        dimensions=(fixture.dim_j,),
        levels=(fixture.builder.level(LevelKind.DENSE, 0),),
    )
    expect_defect("relayout_operand_mismatch", fixture.program)

    # A compressed level is outside the family.
    fixture = build_relayout_spmm()
    operand_decl = fixture.program.tensors[1]
    forge(
        operand_decl,
        levels=(
            fixture.builder.level(LevelKind.DENSE, 0),
            fixture.builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    expect_defect("relayout_operand_mismatch", fixture.program)

    # The last storage level must store the pack dimension.  (The panel's
    # own bound consistency is checked first on the shared operand, so the
    # forge changes the mode-1 dimension, not the level order.)
    fixture = build_relayout_spmm()
    operand_decl = fixture.program.tensors[1]
    forge(operand_decl, dimensions=(fixture.dim_j, fixture.dim_j))
    expect_defect("relayout_operand_mismatch", fixture.program)

    # The operand must be a declared input.
    fixture = build_relayout_spmm()
    forge(fixture.decl, operand=fixture.builder.new_symbol_id())
    expect_defect("undefined_tensor", fixture.program)
    fixture = build_relayout_spmm()
    forge(fixture.decl, operand=fixture.c)
    expect_defect("output_read", fixture.program)


def test_staged_read_binder_discipline():
    # The row index must be the panel's window coordinate.
    fixture = build_relayout_spmm()
    forge(
        fixture.staged,
        indices=(
            fixture.builder.index_value(fixture.row),
            fixture.builder.index_value(fixture.free),
        ),
    )
    expect_defect("domain_mismatch", fixture.program)

    # A coordinate of the right dimension bound by the wrong loop still
    # fails: the row must be the window coordinate itself.
    fixture = build_relayout_spmm()
    inner_row = fixture.builder.new_index_id()
    rebind = fixture.builder.dense_for(
        inner_row, fixture.dim_j, fixture.builder.block((fixture.leaf,))
    )
    forge(
        fixture.staged,
        indices=(
            fixture.builder.index_value(inner_row),
            fixture.builder.index_value(fixture.free),
        ),
    )
    forge(fixture.pack_point, body=fixture.builder.block((rebind,)))
    expect_defect("relayout_read_mismatch", fixture.program)

    # The column index must be the pack split's point coordinate.
    fixture = build_relayout_spmm()
    forge(
        fixture.staged,
        indices=(
            fixture.builder.index_value(fixture.col),
            fixture.builder.index_value(fixture.col),
        ),
    )
    expect_defect("domain_mismatch", fixture.program)

    # A non-IndexValue coordinate expression is rejected even when its
    # domain checks out.
    fixture = build_relayout_spmm()
    forge(
        fixture.staged,
        indices=(fixture.window.cursor.parent.coord, fixture.staged.indices[1]),
    )
    expect_defect("shared_node", fixture.program)

    # Arity is the operand's rank.
    fixture = build_relayout_spmm()
    forge(fixture.staged, indices=(fixture.builder.index_value(fixture.col),))
    expect_defect("rank_mismatch", fixture.program)
    fixture = build_relayout_spmm()
    forge(fixture.staged, indices=[])
    expect_defect("malformed_state", fixture.program)


def test_staged_read_outside_its_panel_window_fails():
    # A staged read lexically inside the region but outside the window has
    # no window coordinate to read.
    fixture = build_relayout_spmm(RelayoutScope.PACK_AXIS)
    outside = fixture.builder.staged_read(
        fixture.relayout,
        (
            fixture.builder.index_value(fixture.col),
            fixture.builder.index_value(fixture.free),
        ),
    )
    outside_point = fixture.builder.tile_inner_for(
        fixture.pack_tile,
        fixture.free,
        fixture.dim_k,
        4,
        False,
        fixture.builder.block(
            (
                fixture.builder.store_reduce(
                    fixture.c,
                    (
                        fixture.builder.index_value(fixture.row),
                        fixture.builder.index_value(fixture.free),
                    ),
                    ReduceOp.ADD,
                    outside,
                ),
            )
        ),
    )
    row_loop = fixture.builder.dense_for(
        fixture.row, fixture.dim_i, fixture.builder.block((outside_point,))
    )
    forge(fixture.stage, body=fixture.builder.block((row_loop,)))
    expect_defect("relayout_read_mismatch", fixture.program)


def test_relayout_region_must_be_read():
    fixture = build_relayout_spmm()
    forge(
        fixture.leaf,
        value=fixture.builder.cursor_value(fixture.window.cursor.cursor),
    )
    expect_defect("relayout_dead_region", fixture.program)


def test_relayout_forged_state_fails_closed():
    # A missing stored field on the decl fails before any checker reads it.
    fixture = build_relayout_spmm()
    object.__delattr__(fixture.decl, "scope")
    expect_defect("malformed_state", fixture.program)

    # A non-member scope is rejected exactly.
    class HostileScope:
        value = "panel"

    fixture = build_relayout_spmm()
    forge(fixture.decl, scope=HostileScope())
    expect_defect("malformed_state", fixture.program)

    # A hostile RelayoutStage subclass is not a registered statement.
    class HostileStage(RelayoutStage):
        pass

    fixture = build_relayout_spmm(RelayoutScope.PACK_AXIS)
    hostile = HostileStage(LoopIRNodeId(10_005), fixture.decl, fixture.stage.body)
    forge(fixture.pack, body=fixture.builder.block((hostile,)))
    expect_defect("unknown_stmt", fixture.program)

    # A hostile StagedRead subclass is not a registered expression.
    class HostileRead(StagedRead):
        pass

    fixture = build_relayout_spmm()
    hostile_read = HostileRead(
        LoopIRNodeId(10_006), fixture.relayout, fixture.staged.indices
    )
    forge(fixture.leaf, value=hostile_read)
    expect_defect("unknown_expr", fixture.program)

    # A decl that is not an exact RelayoutDecl fails closed.
    fixture = build_relayout_spmm()
    forge(fixture.stage, decl=object())
    expect_defect("malformed_state", fixture.program)

    # Cyclic region bodies are caught by the shared structure guards.
    fixture = build_relayout_spmm()
    forge(fixture.stage, body=fixture.program.body)
    expect_defect("cyclic_structure", fixture.program)


def test_relayout_defect_paths_are_reported():
    fixture = build_relayout_spmm()
    forge(fixture.decl, scope=RelayoutScope.PACK_AXIS)
    defect = expect_defect("relayout_scope_mismatch", fixture.program)
    assert "statements[0]" in defect.path


# -- Phase-6 heap result-tile subset ------------------------------------------


@dataclasses.dataclass
class HeapSpmmFixture:
    builder: LoopIRBuilder
    program: LoopProgram
    a: SymbolId
    b: SymbolId
    c: SymbolId
    dim_i: DimensionId
    dim_j: DimensionId
    dim_k: DimensionId
    row: IndexId
    col: IndexId
    free: IndexId
    pack_tile: TileId
    result_tile: ResultTileId
    decl: object
    region: ResultTileRegion
    pack: TileOuterFor
    row_loop: DenseFor
    sparse: Stmt
    pack_point: TileInnerFor
    leaf: TiledReduce


def build_heap_spmm(strip=4, dtype=ScalarType.FLOAT32) -> HeapSpmmFixture:
    """C[i, k] += A[i, j] * B[j, k] with a heap-backed compact result tile.

    An outermost affine pack tile over ``k`` whose whole body is one
    :class:`ResultTileRegion`; the compute leaf accumulates through
    :class:`TiledReduce` instead of a direct result reduction.
    """

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    dim_k = builder.dimension("k")
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    decl_a = builder.tensor(
        a,
        "A",
        dtype,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.DENSE, 0),
            builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    decl_b = builder.tensor(
        b, "B", dtype, (dim_j.dimension, dim_k.dimension), builder.dense_levels(2)
    )
    decl_c = builder.tensor(
        c, "C", dtype, (dim_i.dimension, dim_k.dimension), builder.dense_levels(2)
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    free = builder.new_index_id()
    cursor = builder.new_cursor_id()
    position = builder.new_position_id()
    pack_tile = builder.new_tile_id()
    result_tile = builder.new_result_tile_id()
    cursor_decl = builder.sparse_cursor(
        cursor,
        a,
        1,
        builder.dense_position(a, 0, builder.root_position(), builder.index_value(row)),
    )
    leaf = builder.tiled_reduce(
        result_tile,
        (builder.index_value(row), builder.index_value(free)),
        ReduceOp.ADD,
        builder.binary(
            BinaryOp.MUL,
            builder.cursor_value(cursor),
            builder.load(b, (builder.index_value(col), builder.index_value(free))),
        ),
    )
    pack_point = builder.tile_inner_for(
        pack_tile, free, dim_k.dimension, strip, False, builder.block((leaf,))
    )
    sparse = builder.sparse_for(
        cursor_decl, position, col, builder.block((pack_point,))
    )
    row_loop = builder.dense_for(row, dim_i.dimension, builder.block((sparse,)))
    decl = builder.result_tile_decl(result_tile, c, pack_tile)
    region = builder.result_tile_region(decl, builder.block((row_loop,)))
    pack = builder.tile_outer_for(
        pack_tile, free, dim_k.dimension, strip, builder.block((region,))
    )
    program = builder.program(
        (dim_i, dim_j, dim_k),
        (decl_a, decl_b, decl_c),
        (a, b),
        (c,),
        builder.block((pack,)),
    )
    return HeapSpmmFixture(
        builder,
        program,
        a,
        b,
        c,
        dim_i.dimension,
        dim_j.dimension,
        dim_k.dimension,
        row,
        col,
        free,
        pack_tile,
        result_tile,
        decl,
        region,
        pack,
        row_loop,
        sparse,
        pack_point,
        leaf,
    )


@dataclasses.dataclass
class HeapTtmFixture:
    builder: LoopIRBuilder
    program: LoopProgram
    core: SymbolId
    factor: SymbolId
    projected: SymbolId
    dim_a: DimensionId
    dim_b: DimensionId
    dim_c: DimensionId
    dim_d: DimensionId
    batch: IndexId
    row: IndexId
    red: IndexId
    free: IndexId
    pack_tile: TileId
    result_tile: ResultTileId
    decl: object
    region: ResultTileRegion
    pack: TileOuterFor
    batch_loop: DenseFor
    row_loop: DenseFor
    sparse: Stmt
    pack_point: TileInnerFor
    leaf: TiledReduce


def build_heap_ttm(strip=3, dtype=ScalarType.FLOAT32) -> HeapTtmFixture:
    """``Projected[a, b, d] += Core[a, b, c] * Factor[c, d]`` on a heap tile.

    The multi-prefix representative of the heap result-tile family: the
    compacted result is rank-3 all-dense, so the tile's dense prefix is the
    two logical axes ``a`` and ``b`` and the compact tile linearizes both.
    Mirrors the retained legacy ``heap_ttm_multi_prefix`` audit golden
    (``Core`` is ``dds``, so the reduction is the sparse level-2 loop).
    """

    builder = LoopIRBuilder()
    dim_a = builder.dimension("a")
    dim_b = builder.dimension("b")
    dim_c = builder.dimension("c")
    dim_d = builder.dimension("d")
    core, factor, projected = (builder.new_symbol_id() for _ in range(3))
    decl_core = builder.tensor(
        core,
        "Core",
        dtype,
        (dim_a.dimension, dim_b.dimension, dim_c.dimension),
        (
            builder.level(LevelKind.DENSE, 0),
            builder.level(LevelKind.DENSE, 1),
            builder.level(LevelKind.COMPRESSED, 2),
        ),
    )
    decl_factor = builder.tensor(
        factor,
        "Factor",
        dtype,
        (dim_c.dimension, dim_d.dimension),
        builder.dense_levels(2),
    )
    decl_projected = builder.tensor(
        projected,
        "Projected",
        dtype,
        (dim_a.dimension, dim_b.dimension, dim_d.dimension),
        builder.dense_levels(3),
    )
    batch = builder.new_index_id()
    row = builder.new_index_id()
    red = builder.new_index_id()
    free = builder.new_index_id()
    cursor = builder.new_cursor_id()
    position = builder.new_position_id()
    pack_tile = builder.new_tile_id()
    result_tile = builder.new_result_tile_id()
    cursor_decl = builder.sparse_cursor(
        cursor,
        core,
        2,
        builder.dense_position(
            core,
            1,
            builder.dense_position(
                core, 0, builder.root_position(), builder.index_value(batch)
            ),
            builder.index_value(row),
        ),
    )
    leaf = builder.tiled_reduce(
        result_tile,
        (
            builder.index_value(batch),
            builder.index_value(row),
            builder.index_value(free),
        ),
        ReduceOp.ADD,
        builder.binary(
            BinaryOp.MUL,
            builder.cursor_value(cursor),
            builder.load(factor, (builder.index_value(red), builder.index_value(free))),
        ),
    )
    pack_point = builder.tile_inner_for(
        pack_tile, free, dim_d.dimension, strip, False, builder.block((leaf,))
    )
    sparse = builder.sparse_for(
        cursor_decl, position, red, builder.block((pack_point,))
    )
    row_loop = builder.dense_for(row, dim_b.dimension, builder.block((sparse,)))
    batch_loop = builder.dense_for(batch, dim_a.dimension, builder.block((row_loop,)))
    decl = builder.result_tile_decl(result_tile, projected, pack_tile)
    region = builder.result_tile_region(decl, builder.block((batch_loop,)))
    pack = builder.tile_outer_for(
        pack_tile, free, dim_d.dimension, strip, builder.block((region,))
    )
    program = builder.program(
        (dim_a, dim_b, dim_c, dim_d),
        (decl_core, decl_factor, decl_projected),
        (core, factor),
        (projected,),
        builder.block((pack,)),
    )
    return HeapTtmFixture(
        builder,
        program,
        core,
        factor,
        projected,
        dim_a.dimension,
        dim_b.dimension,
        dim_c.dimension,
        dim_d.dimension,
        batch,
        row,
        red,
        free,
        pack_tile,
        result_tile,
        decl,
        region,
        pack,
        batch_loop,
        row_loop,
        sparse,
        pack_point,
        leaf,
    )


def test_heap_fixture_verifies():
    verify_program(build_heap_spmm().program)
    verify_program(build_heap_spmm(strip=1).program)
    verify_program(build_heap_spmm(dtype=ScalarType.FLOAT64).program)
    verify_program(build_heap_spmm(strip=MAX_LOOPIR_TILE_WIDTH).program)


def test_multi_prefix_heap_fixture_verifies():
    verify_program(build_heap_ttm().program)
    verify_program(build_heap_ttm(strip=1).program)
    verify_program(build_heap_ttm(dtype=ScalarType.FLOAT64).program)
    verify_program(build_heap_ttm(strip=MAX_LOOPIR_TILE_WIDTH).program)


def test_result_tile_identity_is_typed_and_unique():
    fixture = build_heap_spmm()
    forge(fixture.decl, result_tile=7)
    expect_defect("invalid_result_tile_id", fixture.program)

    fixture = build_heap_spmm()
    forge(fixture.leaf, result_tile="0")
    expect_defect("invalid_result_tile_id", fixture.program)

    class HostileResultTileId(ResultTileId):
        pass

    fixture = build_heap_spmm()
    forge(fixture.decl, result_tile=HostileResultTileId(fixture.result_tile.value))
    expect_defect("invalid_result_tile_id", fixture.program)

    fixture = build_heap_spmm()
    forge(fixture.decl, result_tile=ResultTileId(True))
    expect_defect("invalid_result_tile_id", fixture.program)

    # A second region may not reuse an already-declared identity.
    fixture = build_heap_spmm()
    inner_decl = fixture.builder.result_tile_decl(
        fixture.result_tile, fixture.c, fixture.pack_tile
    )
    inner_region = fixture.builder.result_tile_region(inner_decl, fixture.region.body)
    forge(fixture.region, body=fixture.builder.block((inner_region,)))
    expect_defect("duplicate_result_tile_id", fixture.program)


def test_tiled_reduce_requires_an_enclosing_region():
    # The reduce sits under the pack origin but no region is open.
    fixture = build_heap_spmm()
    forge(fixture.pack, body=fixture.region.body)
    expect_defect("unbound_result_tile", fixture.program)

    # A next sibling after the region exits may not reuse its identity.
    fixture = build_heap_spmm()
    stray = fixture.builder.tiled_reduce(
        fixture.result_tile,
        (
            fixture.builder.index_value(fixture.row),
            fixture.builder.index_value(fixture.free),
        ),
        ReduceOp.ADD,
        fixture.builder.float_const(1.0),
    )
    local_region = fixture.builder.result_tile_region(
        fixture.decl, fixture.builder.block((fixture.leaf,))
    )
    forge(fixture.pack_point, body=fixture.builder.block((local_region, stray)))
    forge(fixture.pack, body=fixture.region.body)
    # The nested region now opens inside its own split's point loop.
    expect_defect("result_tile_scope_mismatch", fixture.program)

    fixture = build_heap_spmm()
    huge = ResultTileId(10**5000)
    forge(fixture.leaf, result_tile=huge)
    defect = expect_defect("unbound_result_tile", fixture.program)
    assert "too large" in defect.message


def test_result_tile_region_scope_discipline():
    # The pack split's origin loop must be open.
    fixture = build_heap_spmm()
    forge(fixture.decl, pack=fixture.builder.new_tile_id())
    expect_defect("result_tile_scope_mismatch", fixture.program)

    # A region inside its own split's point loop can never accumulate.
    fixture = build_heap_spmm()
    inner_region = fixture.builder.result_tile_region(
        fixture.decl, fixture.builder.block((fixture.leaf,))
    )
    forge(fixture.pack_point, body=fixture.builder.block((inner_region,)))
    forge(fixture.pack, body=fixture.builder.block((fixture.row_loop,)))
    forge(fixture.sparse, body=fixture.builder.block((fixture.pack_point,)))
    expect_defect("result_tile_scope_mismatch", fixture.program)

    # A region below the row loop resets and copies the complete row-prefix
    # space once per row.  That shape used to verify even though the oracle
    # demonstrably erased rows that were copied by earlier iterations.
    fixture = build_heap_spmm()
    inner_region = fixture.builder.result_tile_region(
        fixture.decl, fixture.row_loop.body
    )
    forge(fixture.row_loop, body=fixture.builder.block((inner_region,)))
    forge(fixture.pack, body=fixture.builder.block((fixture.row_loop,)))
    expect_defect("result_tile_scope_mismatch", fixture.program)

    # Nor may the pack/region pair itself repeat under another loop: region
    # entry and copy-out are once-per-pack-origin whole-prefix semantics.
    fixture = build_heap_spmm()
    wrapper = fixture.builder.dense_for(
        fixture.builder.new_index_id(),
        fixture.dim_i,
        fixture.builder.block((fixture.pack,)),
    )
    forge(fixture.program, body=fixture.builder.block((wrapper,)))
    expect_defect("result_tile_scope_mismatch", fixture.program)

    # Nested regions of one result conflict at copy-out.
    fixture = build_heap_spmm()
    inner_decl = fixture.builder.result_tile_decl(
        fixture.builder.new_result_tile_id(), fixture.c, fixture.pack_tile
    )
    inner_region = fixture.builder.result_tile_region(inner_decl, fixture.region.body)
    forge(fixture.region, body=fixture.builder.block((inner_region,)))
    expect_defect("result_tile_scope_mismatch", fixture.program)


def test_result_tile_result_structure_is_verified():
    # The accumulated tensor must be a declared output.
    fixture = build_heap_spmm()
    forge(fixture.decl, result=fixture.b)
    expect_defect("result_tile_result_mismatch", fixture.program)

    # A rank-one result has no dense prefix to compact.
    fixture = build_heap_spmm()
    forge(
        fixture.program.tensors[2],
        levels=(fixture.builder.level(LevelKind.DENSE, 0),),
        dimensions=(fixture.dim_k,),
    )
    forge(fixture.leaf, indices=(fixture.builder.index_value(fixture.free),))
    expect_defect("result_tile_result_mismatch", fixture.program)

    # A compressed result level is outside the compact family.
    fixture = build_heap_spmm()
    forge(
        fixture.program.tensors[2],
        levels=(
            fixture.builder.level(LevelKind.DENSE, 0),
            fixture.builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    expect_defect("result_tile_result_mismatch", fixture.program)

    # The trailing storage level must store the pack split's dimension.
    fixture = build_heap_spmm()
    forge(
        fixture.program.tensors[2],
        levels=(
            fixture.builder.level(LevelKind.DENSE, 1),
            fixture.builder.level(LevelKind.DENSE, 0),
        ),
    )
    expect_defect("result_tile_result_mismatch", fixture.program)


def test_tiled_reduce_binder_discipline():
    # The trailing index must be the pack split's point coordinate; the
    # sparse reduction coordinate iterates another dimension entirely.
    fixture = build_heap_spmm()
    forge(
        fixture.leaf,
        indices=(
            fixture.builder.index_value(fixture.row),
            fixture.builder.index_value(fixture.col),
        ),
    )
    expect_defect("domain_mismatch", fixture.program)

    # A dense-bound rebinding of the same dimension is rejected exactly.
    fixture = build_heap_spmm()
    rebound = fixture.builder.new_index_id()
    dense_k = fixture.builder.dense_for(
        rebound, fixture.dim_k, fixture.builder.block((fixture.leaf,))
    )
    forge(
        fixture.leaf,
        indices=(
            fixture.builder.index_value(fixture.row),
            fixture.builder.index_value(rebound),
        ),
    )
    forge(fixture.pack_point, body=fixture.builder.block((dense_k,)))
    expect_defect("result_tile_write_mismatch", fixture.program)

    # A non-IndexValue trailing index is rejected.
    fixture = build_heap_spmm()
    forge(
        fixture.leaf,
        indices=(
            fixture.builder.index_value(fixture.row),
            fixture.builder.binary(
                BinaryOp.ADD,
                fixture.builder.index_value(fixture.free),
                fixture.builder.index_value(fixture.free),
            ),
        ),
    )
    expect_defect("type_mismatch", fixture.program)

    # Rank and op discipline.
    fixture = build_heap_spmm()
    forge(fixture.leaf, indices=(fixture.builder.index_value(fixture.row),))
    expect_defect("rank_mismatch", fixture.program)

    fixture = build_heap_spmm()
    forge(fixture.leaf, op="add")
    expect_defect("malformed_state", fixture.program)

    fixture = build_heap_spmm()
    forge(fixture.leaf, indices=[fixture.builder.index_value(fixture.row)])
    expect_defect("malformed_state", fixture.program)


def test_result_tile_rejects_residual_direct_writes():
    fixture = build_heap_spmm()
    residual = fixture.builder.store_reduce(
        fixture.c,
        (
            fixture.builder.index_value(fixture.row),
            fixture.builder.index_value(fixture.free),
        ),
        ReduceOp.ADD,
        fixture.builder.float_const(1.0),
    )
    forge(fixture.pack_point, body=fixture.builder.block((fixture.leaf, residual)))
    expect_defect("result_tile_residual_write", fixture.program)


def test_result_tile_region_must_accumulate():
    fixture = build_heap_spmm()
    plain = fixture.builder.store_reduce(
        fixture.c,
        (
            fixture.builder.index_value(fixture.row),
            fixture.builder.index_value(fixture.free),
        ),
        ReduceOp.ADD,
        fixture.builder.float_const(1.0),
    )
    # Replace the tiled reduce with a direct write outside any region: the
    # region then never accumulates.
    forge(fixture.pack_point, body=fixture.builder.block((plain,)))
    expect_defect("result_tile_residual_write", fixture.program)

    fixture = build_heap_spmm()
    forge(
        fixture.pack_point,
        body=fixture.builder.block(
            (
                fixture.builder.workspace_reduce(
                    fixture.builder.new_workspace_id(),
                    fixture.builder.index_value(fixture.free),
                    ReduceOp.ADD,
                    fixture.builder.float_const(0.0),
                ),
            )
        ),
    )
    expect_defect("unbound_workspace", fixture.program)


def test_result_tile_dead_region_is_rejected():
    fixture = build_heap_spmm()
    stray_output = fixture.builder.new_symbol_id()
    # An empty-bodied region (no reduce anywhere) is dead.
    del stray_output
    forge(
        fixture.pack_point,
        body=fixture.builder.block(()),
    )
    expect_defect("result_tile_dead_region", fixture.program)


def test_result_tile_forged_state_fails_closed():
    # A missing stored field on the decl fails before any checker reads it.
    fixture = build_heap_spmm()
    object.__delattr__(fixture.decl, "pack")
    expect_defect("malformed_state", fixture.program)

    # A decl that is not an exact ResultTileDecl fails closed.
    fixture = build_heap_spmm()
    forge(fixture.region, decl=object())
    expect_defect("malformed_state", fixture.program)

    # A hostile ResultTileRegion subclass is not a registered statement.
    class HostileRegion(ResultTileRegion):
        pass

    fixture = build_heap_spmm()
    hostile = HostileRegion(LoopIRNodeId(10_007), fixture.decl, fixture.region.body)
    forge(fixture.pack, body=fixture.builder.block((hostile,)))
    expect_defect("unknown_stmt", fixture.program)

    # A hostile TiledReduce subclass is not a registered statement.
    class HostileReduce(TiledReduce):
        pass

    fixture = build_heap_spmm()
    hostile_reduce = HostileReduce(
        LoopIRNodeId(10_008),
        fixture.result_tile,
        fixture.leaf.indices,
        ReduceOp.ADD,
        fixture.builder.float_const(0.0),
    )
    forge(fixture.pack_point, body=fixture.builder.block((hostile_reduce,)))
    expect_defect("unknown_stmt", fixture.program)

    # Cyclic region bodies are caught by the shared structure guards.
    fixture = build_heap_spmm()
    forge(fixture.region, body=fixture.program.body)
    expect_defect("cyclic_structure", fixture.program)


def test_result_tile_defect_paths_are_reported():
    fixture = build_heap_spmm()
    forge(fixture.decl, pack=fixture.builder.new_tile_id())
    defect = expect_defect("result_tile_scope_mismatch", fixture.program)
    assert "statements[0]" in defect.path


# -- Phase-6 abstract parallel selection --------------------------------------


def attach_selection(
    fixture,
    index,
    *,
    part=ParallelPart.LOGICAL,
    discipline=ParallelDiscipline.RESULT_PARTITION,
    rows,
    nnz=None,
    intent=ParallelIntent.EXPLICIT,
):
    """Stamp one parallel selection onto a fixture's frozen program."""

    work = fixture.builder.parallel_work(rows, nnz)
    selection = fixture.builder.parallel_selection(
        index, part, discipline, work, intent
    )
    forge(fixture.program, parallel=selection)
    return selection


def test_parallel_selection_verifies_result_partition():
    """A row selection over the CSR SpMV chain is race free and exact."""

    fixture = build_csr_spmv()
    dim_i = fixture.program.tensors[2].dimensions[0]
    attach_selection(
        fixture,
        fixture.row,
        rows=dim_i,
        nnz=fixture.builder.sparse_work_source(fixture.a, 1),
    )
    verify_program(fixture.program)
    missing_source = build_csr_spmv()
    dim_i = missing_source.program.tensors[2].dimensions[0]
    attach_selection(missing_source, missing_source.row, rows=dim_i)
    expect_defect("parallel_work_mismatch", missing_source.program)


def test_parallel_selection_verifies_compact_partition():
    """A dense prefix selection inside the heap region partitions the tile."""

    fixture = build_heap_spmm()
    attach_selection(
        fixture,
        fixture.row,
        discipline=ParallelDiscipline.COMPACT_PARTITION,
        rows=fixture.dim_i,
        nnz=fixture.builder.sparse_work_source(fixture.a, 1),
    )
    verify_program(fixture.program)


def test_parallel_selection_verifies_outer_part():
    """An OUTER selection names a split origin; region state stays private."""

    fixture = build_stack_matmul()
    dim_k = fixture.program.tensors[2].dimensions[1]
    attach_selection(fixture, fixture.col, part=ParallelPart.OUTER, rows=dim_k)
    verify_program(fixture.program)


@pytest.mark.parametrize(
    "mutation",
    [
        "not_a_selection",
        "forged_part",
        "string_discipline",
        "string_intent",
        "work_not_typed",
        "rows_not_dimension",
        "rows_undeclared",
        "nnz_not_typed",
        "nnz_undeclared_tensor",
        "nnz_dense_level",
        "nnz_bool_level",
        "nnz_out_of_range_level",
    ],
)
def test_parallel_selection_rejects_malformed_state(mutation):
    fixture = build_csr_spmv()
    dim_i = fixture.program.tensors[2].dimensions[0]
    selection = attach_selection(fixture, fixture.row, rows=dim_i)
    builder = fixture.builder
    if mutation == "not_a_selection":
        forge(fixture.program, parallel=object())
    elif mutation == "forged_part":
        forged = object.__new__(ParallelPart)
        object.__setattr__(forged, "_name_", "LOGICAL")
        object.__setattr__(forged, "_value_", "logical")
        forge(selection, part=forged)
    elif mutation == "string_discipline":
        forge(selection, discipline="result_partition")
    elif mutation == "string_intent":
        forge(selection, intent="explicit")
    elif mutation == "work_not_typed":
        forge(selection, work=object())
    elif mutation == "rows_not_dimension":
        forge(selection.work, rows=7)
        expect_defect("invalid_dimension_id", fixture.program)
        return
    elif mutation == "rows_undeclared":
        forge(selection.work, rows=DimensionId(4096))
    elif mutation == "nnz_not_typed":
        forge(selection.work, nnz=object())
    elif mutation == "nnz_undeclared_tensor":
        from scorch.compiler.identity import new_symbol_id

        forge(selection.work, nnz=builder.sparse_work_source(new_symbol_id(), 1))
    elif mutation == "nnz_dense_level":
        forge(selection.work, nnz=builder.sparse_work_source(fixture.a, 0))
    elif mutation == "nnz_bool_level":
        forge(selection.work, nnz=builder.sparse_work_source(fixture.a, True))
    else:
        assert mutation == "nnz_out_of_range_level"
        forge(selection.work, nnz=builder.sparse_work_source(fixture.a, 2))
    expect_defect("invalid_parallel_selection", fixture.program)


def test_parallel_selection_missing_stored_field_is_malformed():
    fixture = build_csr_spmv()
    dim_i = fixture.program.tensors[2].dimensions[0]
    selection = attach_selection(fixture, fixture.row, rows=dim_i)
    object.__delattr__(selection, "discipline")
    expect_defect("malformed_state", fixture.program)


def test_parallel_selection_deleted_program_field_is_malformed():
    fixture = build_csr_spmv()
    object.__delattr__(fixture.program, "parallel")
    expect_defect("malformed_state", fixture.program)


def test_parallel_selection_target_missing():
    fixture = build_csr_spmv()
    dim_i = fixture.program.tensors[2].dimensions[0]
    attach_selection(fixture, fixture.builder.new_index_id(), rows=dim_i)
    expect_defect("parallel_target_missing", fixture.program)
    outer = build_csr_spmv()
    dim_i = outer.program.tensors[2].dimensions[0]
    attach_selection(outer, outer.row, part=ParallelPart.OUTER, rows=dim_i)
    expect_defect("parallel_target_missing", outer.program)


def test_parallel_selection_identity_is_unique_by_binding_discipline():
    """Sibling rebinding fails first, so a selection target is unique."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    x, y = (builder.new_symbol_id() for _ in range(2))
    decl_x = builder.tensor(
        x, "x", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    decl_y = builder.tensor(
        y, "y", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    index = builder.new_index_id()
    first = builder.dense_for(
        index,
        dim_i.dimension,
        builder.block(
            (
                builder.store(
                    y,
                    (builder.index_value(index),),
                    builder.load(x, (builder.index_value(index),)),
                ),
            )
        ),
    )
    second = builder.dense_for(
        index,
        dim_i.dimension,
        builder.block(
            (
                builder.store(
                    y,
                    (builder.index_value(index),),
                    builder.load(x, (builder.index_value(index),)),
                ),
            )
        ),
    )
    work = builder.parallel_work(dim_i.dimension, None)
    selection = builder.parallel_selection(
        index,
        ParallelPart.LOGICAL,
        ParallelDiscipline.RESULT_PARTITION,
        work,
        ParallelIntent.EXPLICIT,
    )
    program = builder.program(
        (dim_i,),
        (decl_x, decl_y),
        (x,),
        (y,),
        builder.block((first, second)),
        selection,
    )
    expect_defect("duplicate_index_binding", program)


def test_parallel_selection_work_rows_mismatch():
    fixture = build_csr_spmv()
    dim_j = fixture.program.tensors[1].dimensions[0]
    attach_selection(fixture, fixture.row, rows=dim_j)
    expect_defect("parallel_work_mismatch", fixture.program)


def test_parallel_selection_work_nnz_not_iterated():
    """A sparse work source outside the selected loop is a false estimate."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, x, y, z = (builder.new_symbol_id() for _ in range(4))
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
    decl_z = builder.tensor(
        z, "z", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    other = builder.new_index_id()
    cursor = builder.new_cursor_id()
    position = builder.new_position_id()
    cursor_decl = builder.sparse_cursor(
        cursor,
        a,
        1,
        builder.dense_position(a, 0, builder.root_position(), builder.index_value(row)),
    )
    spmv_leaf = builder.store_reduce(
        y,
        (builder.index_value(row),),
        ReduceOp.ADD,
        builder.binary(
            BinaryOp.MUL,
            builder.cursor_value(cursor),
            builder.load(x, (builder.index_value(col),)),
        ),
    )
    spmv_chain = builder.dense_for(
        row,
        dim_i.dimension,
        builder.block(
            (
                builder.sparse_for(
                    cursor_decl, position, col, builder.block((spmv_leaf,))
                ),
            )
        ),
    )
    dense_chain = builder.dense_for(
        other,
        dim_i.dimension,
        builder.block(
            (
                builder.store(
                    z,
                    (builder.index_value(other),),
                    builder.float_const(1.0),
                ),
            )
        ),
    )
    work = builder.parallel_work(dim_i.dimension, builder.sparse_work_source(a, 1))
    selection = builder.parallel_selection(
        other,
        ParallelPart.LOGICAL,
        ParallelDiscipline.RESULT_PARTITION,
        work,
        ParallelIntent.EXPLICIT,
    )
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_x, decl_y, decl_z),
        (a, x),
        (y, z),
        builder.block((spmv_chain, dense_chain)),
        selection,
    )
    expect_defect("parallel_work_mismatch", program)


def test_parallel_work_uses_the_first_sparse_sibling_in_source_order():
    """The canonical work source follows emitted statement order, not LIFO."""

    fixture = build_csr_spmv()
    builder = fixture.builder
    decl_a = fixture.program.tensors[0]
    b = builder.new_symbol_id()
    z = builder.new_symbol_id()
    decl_b = builder.tensor(
        b,
        "B",
        ScalarType.FLOAT32,
        decl_a.dimensions,
        (
            builder.level(LevelKind.DENSE, 0),
            builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    decl_z = builder.tensor(
        z,
        "z",
        ScalarType.FLOAT32,
        (decl_a.dimensions[0],),
        builder.dense_levels(1),
    )
    col = builder.new_index_id()
    cursor = builder.new_cursor_id()
    cursor_decl = builder.sparse_cursor(
        cursor,
        b,
        1,
        builder.dense_position(
            b,
            0,
            builder.root_position(),
            builder.index_value(fixture.row),
        ),
    )
    second = builder.sparse_for(
        cursor_decl,
        builder.new_position_id(),
        col,
        builder.block(
            (
                builder.store_reduce(
                    z,
                    (builder.index_value(fixture.row),),
                    ReduceOp.ADD,
                    builder.cursor_value(cursor),
                ),
            )
        ),
    )
    outer = fixture.program.body.statements[0]
    assert type(outer) is DenseFor
    forge(outer.body, statements=outer.body.statements + (second,))
    forge(
        fixture.program,
        tensors=fixture.program.tensors + (decl_b, decl_z),
        inputs=fixture.program.inputs + (b,),
        outputs=fixture.program.outputs + (z,),
    )
    selection = attach_selection(
        fixture,
        fixture.row,
        rows=decl_a.dimensions[0],
        nnz=builder.sparse_work_source(fixture.a, 1),
    )
    verify_program(fixture.program)
    forge(selection.work, nnz=builder.sparse_work_source(b, 1))
    expect_defect("parallel_work_mismatch", fixture.program)


def test_parallel_selection_rejects_unrepresented_sparse_trip_count():
    """Sparse position loops need a richer trip-count representation."""

    fixture = build_csr_spmv()
    dim_j = fixture.program.tensors[1].dimensions[0]
    attach_selection(fixture, fixture.col, rows=dim_j)
    expect_defect("parallel_work_mismatch", fixture.program)


def test_parallel_selection_rejects_ordered_assembly():
    fixture = build_union_add()
    dim_i = fixture.program.tensors[0].dimensions[0]
    attach_selection(fixture, fixture.row, rows=dim_i)
    expect_defect("parallel_race", fixture.program)


def test_parallel_selection_rejects_shared_workspace():
    """A loop inside the producer shares the workspace across iterations."""

    fixture = build_stack_matmul()
    dim_j = fixture.program.tensors[0].dimensions[1]
    attach_selection(fixture, fixture.red, rows=dim_j)
    expect_defect("parallel_race", fixture.program)


def test_parallel_selection_result_partition_rejects_heap_regions():
    fixture = build_heap_spmm()
    attach_selection(
        fixture,
        fixture.row,
        rows=fixture.dim_i,
        nnz=fixture.builder.sparse_work_source(fixture.a, 1),
    )
    expect_defect("parallel_race", fixture.program)


def test_parallel_selection_compact_requires_the_heap_region():
    fixture = build_csr_spmv()
    dim_i = fixture.program.tensors[2].dimensions[0]
    attach_selection(
        fixture,
        fixture.row,
        discipline=ParallelDiscipline.COMPACT_PARTITION,
        rows=dim_i,
        nnz=fixture.builder.sparse_work_source(fixture.a, 1),
    )
    expect_defect("parallel_race", fixture.program)


def test_parallel_selection_compact_rejects_the_shared_pack_origin():
    """The pack origin lies outside the region and shares compact storage."""

    fixture = build_heap_spmm()
    attach_selection(
        fixture,
        fixture.free,
        part=ParallelPart.OUTER,
        discipline=ParallelDiscipline.COMPACT_PARTITION,
        rows=fixture.dim_k,
    )
    expect_defect("parallel_race", fixture.program)


def test_parallel_selection_compact_rejects_an_unaddressed_loop():
    """A sparse reduction loop lacks a representable position trip count."""

    fixture = build_heap_spmm()
    attach_selection(
        fixture,
        fixture.col,
        discipline=ParallelDiscipline.COMPACT_PARTITION,
        rows=fixture.dim_j,
    )
    expect_defect("parallel_work_mismatch", fixture.program)


# -- Phase-7 mixed dense-leaf operand loads -----------------------------------


@dataclasses.dataclass
class MixedLeafOperandFixture:
    builder: LoopIRBuilder
    program: LoopProgram
    a: SymbolId
    c: SymbolId
    row: IndexId
    col: IndexId
    load: object


def build_mixed_leaf_operand_copy(forge_load=None) -> MixedLeafOperandFixture:
    """C[i, j] = A[i, j] with A stored compressed-row over a dense leaf."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, c = (builder.new_symbol_id() for _ in range(2))
    decl_a = builder.tensor(
        a,
        "A",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.COMPRESSED, 0),
            builder.level(LevelKind.DENSE, 1),
        ),
    )
    decl_c = builder.tensor(
        c,
        "C",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        builder.dense_levels(2),
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    position = builder.new_position_id()
    cursor_decl = builder.sparse_cursor(
        builder.new_cursor_id(), a, 0, builder.root_position()
    )
    if forge_load is None:
        load = builder.position_load(
            a,
            builder.dense_position(
                a, 1, builder.position_value(position), builder.index_value(col)
            ),
        )
    else:
        load = forge_load(builder, a, c, position, col)
    leaf = builder.store(c, (builder.index_value(row), builder.index_value(col)), load)
    inner = builder.dense_for(col, dim_j.dimension, builder.block((leaf,)))
    outer = builder.sparse_for(cursor_decl, position, row, builder.block((inner,)))
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_c),
        (a,),
        (c,),
        builder.block((outer,)),
    )
    return MixedLeafOperandFixture(builder, program, a, c, row, col, load)


def test_position_load_verifies_through_a_dense_leaf_below_compressed():
    fixture = build_mixed_leaf_operand_copy()
    verify_program(fixture.program)


def test_position_load_rejects_reading_the_output():
    def forge_output_read(builder, a, c, position, col):
        return builder.position_load(
            c,
            builder.dense_position(
                a, 1, builder.position_value(position), builder.index_value(col)
            ),
        )

    fixture = build_mixed_leaf_operand_copy(forge_output_read)
    expect_defect("output_read", fixture.program)


def test_position_load_rejects_an_undeclared_tensor():
    def forge_undeclared(builder, a, c, position, col):
        return builder.position_load(
            SymbolId(999_999),
            builder.dense_position(
                a, 1, builder.position_value(position), builder.index_value(col)
            ),
        )

    fixture = build_mixed_leaf_operand_copy(forge_undeclared)
    expect_defect("undefined_tensor", fixture.program)


def test_position_load_rejects_a_coordinate_typed_position():
    def forge_coordinate(builder, a, c, position, col):
        return builder.position_load(a, builder.index_value(col))

    fixture = build_mixed_leaf_operand_copy(forge_coordinate)
    expect_defect("type_mismatch", fixture.program)


def test_position_load_rejects_another_tensors_position():
    """A leaf position formed on a second input cannot serve this load."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    mixed_levels = lambda: (  # noqa: E731
        builder.level(LevelKind.COMPRESSED, 0),
        builder.level(LevelKind.DENSE, 1),
    )
    decl_a = builder.tensor(
        a, "A", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), mixed_levels()
    )
    decl_b = builder.tensor(
        b, "B", ScalarType.FLOAT32, (dim_i.dimension, dim_j.dimension), mixed_levels()
    )
    decl_c = builder.tensor(
        c,
        "C",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        builder.dense_levels(2),
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    position = builder.new_position_id()
    cursor_decl = builder.sparse_cursor(
        builder.new_cursor_id(), a, 0, builder.root_position()
    )
    load = builder.position_load(
        b,
        builder.dense_position(
            a, 1, builder.position_value(position), builder.index_value(col)
        ),
    )
    leaf = builder.store(c, (builder.index_value(row), builder.index_value(col)), load)
    inner = builder.dense_for(col, dim_j.dimension, builder.block((leaf,)))
    outer = builder.sparse_for(cursor_decl, position, row, builder.block((inner,)))
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_b, decl_c),
        (a, b),
        (c,),
        builder.block((outer,)),
    )
    expect_defect("position_load_mismatch", program)


def test_position_load_rejects_a_structural_level_position():
    """The compressed level-0 position of a rank-2 tensor owns no scalar."""

    def forge_structural(builder, a, c, position, col):
        return builder.position_load(a, builder.position_value(position))

    fixture = build_mixed_leaf_operand_copy(forge_structural)
    expect_defect("non_leaf_value", fixture.program)


# -- rank-K key domains ------------------------------------------------------
#
# ``len(key_dimensions) == 1`` is the K == 1 instance of the same node, not a
# separate kind.  These lock the K >= 2 behaviour the multi-compressed
# reduction/TTM vertical needs: one coordinate and one bound index per key
# dimension, in key order, and a strictly increasing LEXICOGRAPHIC drain.


def build_rank_k_workspace_program(
    key_rank=2,
    *,
    coords=None,
    drain_indices=None,
    key_permutation=None,
    contraction_extent=None,
):
    """One rank-``key_rank`` sparse workspace draining into a rank-K result.

    The region sits ABOVE every producer loop -- the anchoring the
    multi-compressed reduction vertical needs -- so one region accumulates
    the whole key space and drains it once.  ``key_permutation`` reorders the
    key dimensions relative to the producer's loop nest, which is what makes
    the drain's lexicographic contract observable: with a non-identity
    permutation the insertion order is not the key order.

    ``contraction_extent`` adds one innermost producer loop whose index does
    NOT appear in the key -- a contraction.  Every one of its iterations
    inserts the same key, so the region must merge them under ADD into a
    single drained entry.  That is the reduction shape this whole vertical
    exists for.

    ``coords`` / ``drain_indices`` override the insertion key and the drain
    binding so a caller can build a deliberately malformed program.
    """

    builder = LoopIRBuilder()
    loop_dimensions = [builder.dimension(name) for name in "ijk"[:key_rank]]
    order = (
        list(key_permutation) if key_permutation is not None else list(range(key_rank))
    )
    key_dimensions = [loop_dimensions[position] for position in order]
    output = builder.new_symbol_id()
    output_decl = builder.tensor(
        output,
        "C",
        ScalarType.FLOAT32,
        tuple(dimension.dimension for dimension in key_dimensions),
        tuple(builder.level(LevelKind.COMPRESSED, level) for level in range(key_rank)),
    )
    loop_indices = [builder.new_index_id() for _ in range(key_rank)]
    workspace = builder.new_workspace_id()
    workspace_decl = builder.sparse_workspace_decl(
        workspace,
        "wksp",
        ScalarType.FLOAT32,
        tuple(dimension.dimension for dimension in key_dimensions),
    )
    inputs = ()
    inserted_value = builder.float_const(1.0)
    contraction = None
    contraction_index = None
    if contraction_extent is not None:
        # A dense operand over (loop dims..., r) both gives the contraction
        # dimension a tensor-mapped runtime extent and makes the merged
        # value a real reduction with a checkable reference.
        contraction = builder.dimension("r")
        contraction_index = builder.new_index_id()
        operand = builder.new_symbol_id()
        operand_decl = builder.tensor(
            operand,
            "A",
            ScalarType.FLOAT32,
            tuple(dimension.dimension for dimension in loop_dimensions)
            + (contraction.dimension,),
            tuple(
                builder.level(LevelKind.DENSE, level) for level in range(key_rank + 1)
            ),
        )
        inputs = (operand,)
        inserted_value = builder.load(
            operand,
            tuple(builder.index_value(index) for index in loop_indices)
            + (builder.index_value(contraction_index),),
        )
    insert = builder.sparse_workspace_insert(
        workspace,
        (
            coords
            if coords is not None
            else tuple(builder.index_value(loop_indices[p]) for p in order)
        ),
        ReduceOp.ADD,
        inserted_value,
    )
    producer = builder.block((insert,))
    if contraction_extent is not None:
        producer = builder.block(
            (builder.dense_for(contraction_index, contraction.dimension, producer),)
        )
    for index, dimension in zip(reversed(loop_indices), reversed(loop_dimensions)):
        producer = builder.block(
            (builder.dense_for(index, dimension.dimension, producer),)
        )
    bound = (
        drain_indices
        if drain_indices is not None
        else tuple(builder.new_index_id() for _ in range(key_rank))
    )
    append = builder.append_entry(
        output,
        tuple(builder.index_value(index) for index in bound),
        builder.sparse_workspace_value(workspace),
    )
    drain = builder.sparse_workspace_drain_for(
        workspace, bound, builder.block((append,))
    )
    region = builder.sparse_workspace_region(
        workspace_decl, producer, builder.block((drain,))
    )
    program = builder.program(
        tuple(loop_dimensions) + ((contraction,) if contraction is not None else ()),
        ((operand_decl,) if contraction is not None else ()) + (output_decl,),
        inputs,
        (output,),
        builder.block((region,)),
    )
    return builder, program, region, drain


def rank_k_insert(region):
    """The single ``SparseWorkspaceInsert`` at the bottom of the producer."""

    node = region.producer
    while type(node) is not SparseWorkspaceInsert:
        node = node.statements[0] if type(node) is Block else node.body
    return node


@pytest.mark.parametrize("key_rank", [1, 2, 3])
def test_rank_k_sparse_workspace_program_verifies(key_rank):
    _, program, region, drain = build_rank_k_workspace_program(key_rank)
    verify_program(program)
    assert len(region.workspace.key_dimensions) == key_rank
    assert len(drain.indices) == key_rank


def test_sparse_workspace_key_must_be_a_nonempty_tuple():
    builder, program, region, _ = build_rank_k_workspace_program(2)
    forge(region.workspace, key_dimensions=())
    defect = expect_defect("malformed_state", program)
    assert "nonempty tuple" in defect.message


def test_sparse_workspace_key_may_not_repeat_a_dimension():
    builder, program, region, _ = build_rank_k_workspace_program(2)
    repeated = region.workspace.key_dimensions[0]
    forge(region.workspace, key_dimensions=(repeated, repeated))
    defect = expect_defect("malformed_state", program)
    assert "repeat a dimension" in defect.message


def test_insertion_must_supply_one_coordinate_per_key_dimension():
    builder, program, region, _ = build_rank_k_workspace_program(2)
    insert = rank_k_insert(region)
    forge(insert, coords=insert.coords[:1])
    defect = expect_defect("malformed_state", program)
    assert "one coordinate per declared key dimension" in defect.message


def test_drain_must_bind_one_index_per_key_dimension():
    builder, program, region, drain = build_rank_k_workspace_program(2)
    forge(drain, indices=drain.indices[:1])
    defect = expect_defect("malformed_state", program)
    assert "one index per declared key dimension" in defect.message


def test_insertion_coordinates_must_match_their_own_key_dimension():
    """Coordinate agreement is per-position, not merely per-set."""

    builder, program, region, _ = build_rank_k_workspace_program(2)
    insert = rank_k_insert(region)
    forge(insert, coords=(insert.coords[1], insert.coords[0]))
    defect = expect_defect("domain_mismatch", program)
    assert "dimension 'j' but dimension 'i' is required" in defect.message
    assert defect.path.endswith(".coords[0]")
