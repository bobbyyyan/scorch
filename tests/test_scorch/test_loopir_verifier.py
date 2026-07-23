"""Fail-closed coverage for the production LoopIR dense-subset verifier.

Every reachable defect code has at least one direct regression here, built
either through the supported builder API or by forging frozen dataclass state
the way an adversarial caller could.
"""

import dataclasses

import pytest

from scorch.compiler.identity import IndexId, SymbolId
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
    forge(fixture.program.tensors[0].levels[0], kind=LevelKind.COMPRESSED)
    defect = expect_defect("unsupported_level_kind", fixture.program)
    assert "compressed" in defect.message


@pytest.mark.parametrize(
    "kind", [LevelKind.COMPRESSED, LevelKind.COORDINATE, LevelKind.SINGLETON]
)
def test_every_non_dense_level_kind_fails_closed(kind):
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


DENSE_SUBSET_DEFECT_CODES = {
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
}


def test_defect_codes_are_the_documented_dense_subset():
    """Lock the stable defect-code surface of the production verifier."""

    import re

    import scorch.compiler.loopir.verifier as verifier_module

    source = open(verifier_module.__file__).read()
    found = set(re.findall(r"_fail\(\s*\"([a-z_]+)\"", source))
    assert found == DENSE_SUBSET_DEFECT_CODES
