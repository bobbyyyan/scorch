"""Adversarial coverage of every Phase-3.5 LoopIR spike verifier boundary."""

import pytest

from scorch.compiler.identity import new_index_id, new_symbol_id
from scorch.compiler.loopir_spike.nodes import (
    REDUCE_IDENTITIES,
    Accumulate,
    AccumValue,
    AppendEntry,
    BinaryExpr,
    BinaryOp,
    Block,
    CursorValue,
    DeclAccum,
    DenseFor,
    DimSize,
    Expr,
    FloatConst,
    IndexValue,
    IntConst,
    LevelKind,
    Load,
    LoopNodeId,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    ReduceOp,
    SparseCursorDecl,
    SparseFor,
    Stmt,
    Store,
    TensorDecl,
    new_cursor_id,
    new_loop_node_id,
)
from scorch.compiler.loopir_spike.programs import (
    build_csr_intersection_multiply_program,
    build_csr_spmv_program,
    build_csr_union_add_program,
)
from scorch.compiler.loopir_spike.verifier import (
    MAX_NESTING_DEPTH,
    LoopIRVerificationError,
    verify_program,
)

DENSE = LevelKind.DENSE
COMPRESSED = LevelKind.COMPRESSED
CSR = (DENSE, COMPRESSED)


def nid() -> LoopNodeId:
    return new_loop_node_id()


def wrap(tensors, inputs, outputs, stmts) -> LoopProgram:
    return LoopProgram(nid(), tuple(tensors), inputs, outputs, Block(nid(), stmts))


def expect_defect(program, code, path_prefix="program"):
    with pytest.raises(LoopIRVerificationError) as excinfo:
        verify_program(program)
    defect = excinfo.value.defect
    assert defect.code == code, defect
    assert defect.path.startswith(path_prefix), defect
    return defect


class _VecPair:
    """One dense input vector, one dense output vector, and a copy loop."""

    def __init__(self):
        self.x = new_symbol_id()
        self.y = new_symbol_id()
        self.tensors = (
            TensorDecl(nid(), self.x, "x", (DENSE,)),
            TensorDecl(nid(), self.y, "y", (DENSE,)),
        )

    def copy_loop(self):
        i = new_index_id()
        return DenseFor(
            nid(),
            i,
            DimSize(nid(), self.x, 0),
            Block(
                nid(),
                (
                    Store(
                        nid(),
                        self.y,
                        (IndexValue(nid(), i),),
                        Load(nid(), self.x, (IndexValue(nid(), i),)),
                    ),
                ),
            ),
        )

    def program(self, stmts=None):
        statements = (self.copy_loop(),) if stmts is None else stmts
        return wrap(self.tensors, (self.x,), (self.y,), statements)


class _CsrSetup:
    """One CSR input, one CSR output, and merge/cursor building blocks."""

    def __init__(self, second_input=False):
        self.a = new_symbol_id()
        self.c = new_symbol_id()
        tensors = [
            TensorDecl(nid(), self.a, "A", CSR),
            TensorDecl(nid(), self.c, "C", CSR),
        ]
        inputs = [self.a]
        if second_input:
            self.b = new_symbol_id()
            tensors.append(TensorDecl(nid(), self.b, "B", CSR))
            inputs.append(self.b)
        self.tensors = tuple(tensors)
        self.inputs = tuple(inputs)

    def cursor(self, tensor, row_index):
        return SparseCursorDecl(
            node_id=nid(),
            cursor=new_cursor_id(),
            tensor=tensor,
            level=1,
            outer_indices=(IndexValue(nid(), row_index),),
        )

    def program(self, stmts):
        return wrap(self.tensors, self.inputs, (self.c,), stmts)


def test_hand_authored_fixture_programs_verify():
    verify_program(build_csr_spmv_program().program)
    verify_program(build_csr_union_add_program().program)
    verify_program(build_csr_intersection_multiply_program().program)


def test_reduce_identity_table_is_read_only():
    with pytest.raises(TypeError):
        REDUCE_IDENTITIES[ReduceOp.ADD] = 1.0  # type: ignore[index]


def test_fresh_ids_are_unique():
    assert len({new_loop_node_id() for _ in range(64)}) == 64
    assert len({new_cursor_id() for _ in range(64)}) == 64


# ---------------------------------------------------------------- unique IDs


def test_duplicate_node_id_rejected():
    pair = _VecPair()
    shared = nid()
    i = new_index_id()
    loop = DenseFor(
        nid(),
        i,
        DimSize(shared, pair.x, 0),
        Block(
            nid(),
            (
                Store(
                    nid(),
                    pair.y,
                    (IndexValue(shared, i),),
                    Load(nid(), pair.x, (IndexValue(nid(), i),)),
                ),
            ),
        ),
    )
    expect_defect(pair.program((loop,)), "duplicate_node_id")


def test_shared_node_object_rejected():
    pair = _VecPair()
    i = new_index_id()
    shared_index = IndexValue(nid(), i)
    loop = DenseFor(
        nid(),
        i,
        DimSize(nid(), pair.x, 0),
        Block(
            nid(),
            (
                Store(
                    nid(),
                    pair.y,
                    (shared_index,),
                    Load(nid(), pair.x, (shared_index,)),
                ),
            ),
        ),
    )
    expect_defect(pair.program((loop,)), "shared_node")


def test_duplicate_cursor_id_rejected():
    setup = _CsrSetup()
    i = new_index_id()
    first = setup.cursor(setup.a, i)
    second = SparseCursorDecl(
        node_id=nid(),
        cursor=first.cursor,
        tensor=setup.a,
        level=1,
        outer_indices=(IndexValue(nid(), i),),
    )
    j1, j2 = new_index_id(), new_index_id()
    body = Block(
        nid(),
        (
            AppendEntry(
                nid(),
                setup.c,
                (IndexValue(nid(), i), IndexValue(nid(), j2)),
                CursorValue(nid(), second.cursor, None),
            ),
        ),
    )
    inner = SparseFor(nid(), second, j2, body)
    outer = SparseFor(nid(), first, j1, Block(nid(), (inner,)))
    loop = DenseFor(
        nid(),
        i,
        DimSize(nid(), setup.a, 0),
        Block(nid(), (outer,)),
    )
    expect_defect(setup.program((loop,)), "duplicate_cursor_id")


def test_duplicate_accumulator_symbol_rejected():
    pair = _VecPair()
    accumulator = new_symbol_id()
    stmts = (
        Block(
            nid(),
            (DeclAccum(nid(), accumulator, ReduceOp.ADD, FloatConst(nid(), 0.0)),),
        ),
        Block(
            nid(),
            (DeclAccum(nid(), accumulator, ReduceOp.ADD, FloatConst(nid(), 0.0)),),
        ),
        pair.copy_loop(),
    )
    expect_defect(pair.program(stmts), "duplicate_symbol")


def test_accumulator_symbol_colliding_with_tensor_rejected():
    pair = _VecPair()
    stmts = (
        DeclAccum(nid(), pair.x, ReduceOp.ADD, FloatConst(nid(), 0.0)),
        pair.copy_loop(),
    )
    expect_defect(pair.program(stmts), "duplicate_symbol")


def test_duplicate_index_binding_rejected():
    pair = _VecPair()
    i = new_index_id()
    inner = DenseFor(nid(), i, DimSize(nid(), pair.x, 0), Block(nid(), ()))
    outer = DenseFor(
        nid(),
        i,
        DimSize(nid(), pair.x, 0),
        Block(nid(), (inner,)),
    )
    expect_defect(pair.program((outer, pair.copy_loop())), "duplicate_index_binding")


def test_redeclared_tensor_symbol_rejected():
    pair = _VecPair()
    tensors = pair.tensors + (TensorDecl(nid(), pair.x, "x2", (DENSE,)),)
    program = wrap(tensors, (pair.x,), (pair.y,), (pair.copy_loop(),))
    expect_defect(program, "duplicate_symbol", "program.tensors[2]")


# ------------------------------------------------------------- definitions


def test_unbound_index_rejected():
    pair = _VecPair()
    stray = new_index_id()
    stmt = Store(
        nid(),
        pair.y,
        (IndexValue(nid(), stray),),
        FloatConst(nid(), 1.0),
    )
    expect_defect(pair.program((stmt,)), "unbound_index")


def test_cursor_read_after_loop_rejected():
    setup = _CsrSetup()
    i = new_index_id()
    j = new_index_id()
    cursor = setup.cursor(setup.a, i)
    sparse = SparseFor(nid(), cursor, j, Block(nid(), ()))
    late_read = AppendEntry(
        nid(),
        setup.c,
        (IndexValue(nid(), i), IntConst(nid(), 0)),
        CursorValue(nid(), cursor.cursor, None),
    )
    loop = DenseFor(
        nid(),
        i,
        DimSize(nid(), setup.a, 0),
        Block(nid(), (sparse, late_read)),
    )
    expect_defect(setup.program((loop,)), "unbound_cursor")


def test_accum_value_without_declaration_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        AccumValue(nid(), new_symbol_id()),
    )
    expect_defect(pair.program((stmt,)), "undefined_accumulator")


def test_accumulate_before_declaration_rejected():
    pair = _VecPair()
    accumulator = new_symbol_id()
    stmts = (
        Accumulate(nid(), accumulator, FloatConst(nid(), 1.0)),
        DeclAccum(nid(), accumulator, ReduceOp.ADD, FloatConst(nid(), 0.0)),
        pair.copy_loop(),
    )
    expect_defect(pair.program(stmts), "undefined_accumulator")


def test_accumulator_read_after_block_exit_rejected():
    pair = _VecPair()
    accumulator = new_symbol_id()
    stmts = (
        Block(
            nid(),
            (DeclAccum(nid(), accumulator, ReduceOp.ADD, FloatConst(nid(), 0.0)),),
        ),
        Store(
            nid(),
            pair.y,
            (IntConst(nid(), 0),),
            AccumValue(nid(), accumulator),
        ),
    )
    expect_defect(pair.program(stmts), "undefined_accumulator")


def test_load_of_undeclared_tensor_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        Load(nid(), new_symbol_id(), (IntConst(nid(), 0),)),
    )
    expect_defect(pair.program((stmt,)), "undefined_tensor")


def test_dim_size_of_undeclared_tensor_rejected():
    pair = _VecPair()
    i = new_index_id()
    loop = DenseFor(
        nid(),
        i,
        DimSize(nid(), new_symbol_id(), 0),
        Block(nid(), ()),
    )
    expect_defect(pair.program((loop, pair.copy_loop())), "undefined_tensor")


def test_undeclared_input_listing_rejected():
    pair = _VecPair()
    program = wrap(
        pair.tensors,
        (pair.x, new_symbol_id()),
        (pair.y,),
        (pair.copy_loop(),),
    )
    expect_defect(program, "undefined_tensor", "program.inputs[1]")


def test_cursor_outer_index_using_own_coordinate_rejected():
    setup = _CsrSetup()
    j = new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        outer_indices=(IndexValue(nid(), j),),
    )
    sparse = SparseFor(
        nid(),
        decl,
        j,
        Block(
            nid(),
            (
                AppendEntry(
                    nid(),
                    setup.c,
                    (IntConst(nid(), 0), IndexValue(nid(), j)),
                    CursorValue(nid(), decl.cursor, None),
                ),
            ),
        ),
    )
    expect_defect(setup.program((sparse,)), "unbound_index")


# ----------------------------------------------------------- rank and layout


def test_load_from_compressed_tensor_rejected():
    setup = _CsrSetup()
    i = new_index_id()
    loop = DenseFor(
        nid(),
        i,
        DimSize(nid(), setup.a, 0),
        Block(
            nid(),
            (
                AppendEntry(
                    nid(),
                    setup.c,
                    (IndexValue(nid(), i), IntConst(nid(), 0)),
                    Load(
                        nid(),
                        setup.a,
                        (IndexValue(nid(), i), IntConst(nid(), 0)),
                    ),
                ),
            ),
        ),
    )
    expect_defect(setup.program((loop,)), "layout_mismatch")


def test_load_arity_mismatch_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        Load(nid(), pair.x, (IntConst(nid(), 0), IntConst(nid(), 1))),
    )
    expect_defect(pair.program((stmt,)), "rank_mismatch")


def test_cursor_on_dense_level_rejected():
    setup = _CsrSetup()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=0,
        outer_indices=(),
    )
    j = new_index_id()
    sparse = SparseFor(nid(), decl, j, Block(nid(), ()))
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((sparse, append)), "layout_mismatch")


def test_cursor_level_out_of_range_rejected():
    setup = _CsrSetup()
    i = new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=2,
        outer_indices=(IndexValue(nid(), i), IntConst(nid(), 0)),
    )
    j = new_index_id()
    loop = DenseFor(
        nid(),
        i,
        DimSize(nid(), setup.a, 0),
        Block(nid(), (SparseFor(nid(), decl, j, Block(nid(), ())),)),
    )
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((loop, append)), "rank_mismatch")


def test_cursor_outer_arity_mismatch_rejected():
    setup = _CsrSetup()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        outer_indices=(),
    )
    j = new_index_id()
    sparse = SparseFor(nid(), decl, j, Block(nid(), ()))
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((sparse, append)), "rank_mismatch")


def test_store_to_compressed_output_rejected():
    setup = _CsrSetup()
    stmt = Store(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((stmt,)), "layout_mismatch")


def test_append_to_dense_output_rejected():
    pair = _VecPair()
    stmt = AppendEntry(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        FloatConst(nid(), 1.0),
    )
    expect_defect(pair.program((stmt,)), "layout_mismatch")


def test_append_arity_mismatch_rejected():
    setup = _CsrSetup()
    stmt = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0),),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((stmt,)), "rank_mismatch")


def test_store_arity_mismatch_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), 0), IntConst(nid(), 1)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(pair.program((stmt,)), "rank_mismatch")


def test_dim_size_out_of_range_rejected():
    pair = _VecPair()
    i = new_index_id()
    loop = DenseFor(nid(), i, DimSize(nid(), pair.x, 1), Block(nid(), ()))
    expect_defect(pair.program((loop, pair.copy_loop())), "rank_mismatch")


# ------------------------------------------------------------------- typing


def test_value_typed_extent_rejected():
    pair = _VecPair()
    i = new_index_id()
    loop = DenseFor(nid(), i, FloatConst(nid(), 3.0), Block(nid(), ()))
    expect_defect(pair.program((loop, pair.copy_loop())), "type_mismatch")


def test_value_typed_store_index_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (FloatConst(nid(), 0.0),),
        FloatConst(nid(), 1.0),
    )
    expect_defect(pair.program((stmt,)), "type_mismatch")


def test_coordinate_typed_accumulate_rejected():
    pair = _VecPair()
    accumulator = new_symbol_id()
    stmts = (
        DeclAccum(nid(), accumulator, ReduceOp.ADD, FloatConst(nid(), 0.0)),
        Accumulate(nid(), accumulator, IntConst(nid(), 1)),
        pair.copy_loop(),
    )
    expect_defect(pair.program(stmts), "type_mismatch")


def test_coordinate_typed_binary_operand_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        BinaryExpr(
            nid(),
            BinaryOp.ADD,
            FloatConst(nid(), 1.0),
            IntConst(nid(), 1),
        ),
    )
    expect_defect(pair.program((stmt,)), "type_mismatch")


def test_coordinate_typed_union_default_rejected():
    setup = _CsrSetup(second_input=True)
    i = new_index_id()
    j = new_index_id()
    left = setup.cursor(setup.a, i)
    right = setup.cursor(setup.b, i)
    merge = MergedSparseFor(
        nid(),
        MergeMode.UNION,
        (left, right),
        j,
        Block(
            nid(),
            (
                AppendEntry(
                    nid(),
                    setup.c,
                    (IndexValue(nid(), i), IndexValue(nid(), j)),
                    BinaryExpr(
                        nid(),
                        BinaryOp.ADD,
                        CursorValue(nid(), left.cursor, IntConst(nid(), 0)),
                        CursorValue(nid(), right.cursor, FloatConst(nid(), 0.0)),
                    ),
                ),
            ),
        ),
    )
    loop = DenseFor(nid(), i, DimSize(nid(), setup.a, 0), Block(nid(), (merge,)))
    expect_defect(setup.program((loop,)), "type_mismatch")


# ------------------------------------------------------- reduction identity


@pytest.mark.parametrize(
    "op,bad_init",
    [
        (ReduceOp.ADD, 1.0),
        (ReduceOp.MUL, 0.0),
        (ReduceOp.ADD, -0.0),
    ],
)
def test_non_identity_reduction_init_rejected(op, bad_init):
    pair = _VecPair()
    stmts = (
        DeclAccum(nid(), new_symbol_id(), op, FloatConst(nid(), bad_init)),
        pair.copy_loop(),
    )
    expect_defect(pair.program(stmts), "invalid_reduction_identity")


def test_non_literal_reduction_init_rejected():
    pair = _VecPair()
    stmts = (
        DeclAccum(
            nid(),
            new_symbol_id(),
            ReduceOp.ADD,
            BinaryExpr(
                nid(),
                BinaryOp.SUB,
                FloatConst(nid(), 1.0),
                FloatConst(nid(), 1.0),
            ),
        ),
        pair.copy_loop(),
    )
    expect_defect(pair.program(stmts), "invalid_reduction_identity")


def test_mul_reduction_with_its_identity_verifies():
    pair = _VecPair()
    accumulator = new_symbol_id()
    stmts = (
        DeclAccum(nid(), accumulator, ReduceOp.MUL, FloatConst(nid(), 1.0)),
        Accumulate(nid(), accumulator, FloatConst(nid(), 2.0)),
        Store(
            nid(),
            pair.y,
            (IntConst(nid(), 0),),
            AccumValue(nid(), accumulator),
        ),
    )
    verify_program(pair.program(stmts))


# ------------------------------------------------------------- output scope


def test_store_to_input_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.x,
        (IntConst(nid(), 0),),
        FloatConst(nid(), 1.0),
    )
    expect_defect(pair.program((stmt, pair.copy_loop())), "output_scope")


def test_load_from_output_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        Load(nid(), pair.y, (IntConst(nid(), 0),)),
    )
    expect_defect(pair.program((stmt,)), "output_read")


def test_append_to_input_rejected():
    setup = _CsrSetup(second_input=True)
    stmt = AppendEntry(
        nid(),
        setup.b,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((stmt, append)), "output_scope")


def test_cursor_over_output_rejected():
    setup = _CsrSetup()
    i = new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.c,
        level=1,
        outer_indices=(IndexValue(nid(), i),),
    )
    j = new_index_id()
    loop = DenseFor(
        nid(),
        i,
        DimSize(nid(), setup.a, 0),
        Block(nid(), (SparseFor(nid(), decl, j, Block(nid(), ())),)),
    )
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((loop, append)), "output_read")


def test_unwritten_output_rejected():
    pair = _VecPair()
    program = wrap(pair.tensors, (pair.x,), (pair.y,), ())
    expect_defect(program, "unwritten_output", "program.outputs")


def test_roleless_tensor_rejected():
    pair = _VecPair()
    tensors = pair.tensors + (TensorDecl(nid(), new_symbol_id(), "orphan", (DENSE,)),)
    program = wrap(tensors, (pair.x,), (pair.y,), (pair.copy_loop(),))
    expect_defect(program, "output_scope", "program.tensors")


def test_program_without_outputs_rejected():
    pair = _VecPair()
    program = wrap(pair.tensors, (pair.x, pair.y), (), (pair.copy_loop(),))
    expect_defect(program, "output_scope", "program.outputs")


def test_tensor_in_both_roles_rejected():
    pair = _VecPair()
    program = wrap(
        pair.tensors,
        (pair.x,),
        (pair.x, pair.y),
        (pair.copy_loop(),),
    )
    expect_defect(program, "duplicate_symbol", "program.outputs[0]")


# ------------------------------------------------------------ merge boundary


def _merge_program(setup, mode, cursors, body_value):
    i = new_index_id()
    j = new_index_id()
    built = [cursor(i) for cursor in cursors]
    merge = MergedSparseFor(
        nid(),
        mode,
        tuple(built),
        j,
        Block(
            nid(),
            (
                AppendEntry(
                    nid(),
                    setup.c,
                    (IndexValue(nid(), i), IndexValue(nid(), j)),
                    body_value(built, j),
                ),
            ),
        ),
    )
    loop = DenseFor(nid(), i, DimSize(nid(), setup.a, 0), Block(nid(), (merge,)))
    return setup.program((loop,))


def test_single_cursor_merge_rejected():
    setup = _CsrSetup()
    program = _merge_program(
        setup,
        MergeMode.UNION,
        [lambda i: setup.cursor(setup.a, i)],
        lambda built, j: CursorValue(nid(), built[0].cursor, FloatConst(nid(), 0.0)),
    )
    expect_defect(program, "degenerate_merge")


def test_forged_merge_mode_rejected():
    setup = _CsrSetup(second_input=True)
    program = _merge_program(
        setup,
        "union",
        [
            lambda i: setup.cursor(setup.a, i),
            lambda i: setup.cursor(setup.b, i),
        ],
        lambda built, j: FloatConst(nid(), 1.0),
    )
    expect_defect(program, "malformed_state")


def test_union_cursor_read_without_default_rejected():
    setup = _CsrSetup(second_input=True)
    program = _merge_program(
        setup,
        MergeMode.UNION,
        [
            lambda i: setup.cursor(setup.a, i),
            lambda i: setup.cursor(setup.b, i),
        ],
        lambda built, j: CursorValue(nid(), built[0].cursor, None),
    )
    expect_defect(program, "missing_union_default")


def test_intersection_cursor_read_with_default_rejected():
    setup = _CsrSetup(second_input=True)
    program = _merge_program(
        setup,
        MergeMode.INTERSECTION,
        [
            lambda i: setup.cursor(setup.a, i),
            lambda i: setup.cursor(setup.b, i),
        ],
        lambda built, j: CursorValue(nid(), built[0].cursor, FloatConst(nid(), 0.0)),
    )
    expect_defect(program, "dead_default")


def test_sparse_for_cursor_read_with_default_rejected():
    setup = _CsrSetup()
    i = new_index_id()
    j = new_index_id()
    decl = setup.cursor(setup.a, i)
    sparse = SparseFor(
        nid(),
        decl,
        j,
        Block(
            nid(),
            (
                AppendEntry(
                    nid(),
                    setup.c,
                    (IndexValue(nid(), i), IndexValue(nid(), j)),
                    CursorValue(nid(), decl.cursor, FloatConst(nid(), 0.0)),
                ),
            ),
        ),
    )
    loop = DenseFor(nid(), i, DimSize(nid(), setup.a, 0), Block(nid(), (sparse,)))
    expect_defect(setup.program((loop,)), "dead_default")


def test_union_default_reading_a_cursor_rejected():
    setup = _CsrSetup(second_input=True)
    program = _merge_program(
        setup,
        MergeMode.UNION,
        [
            lambda i: setup.cursor(setup.a, i),
            lambda i: setup.cursor(setup.b, i),
        ],
        lambda built, j: CursorValue(
            nid(),
            built[0].cursor,
            CursorValue(nid(), built[1].cursor, FloatConst(nid(), 0.0)),
        ),
    )
    expect_defect(program, "default_contains_cursor")


# ------------------------------------- cycles, malformed state, and depth


def test_forged_cyclic_block_rejected():
    pair = _VecPair()
    block = Block(nid(), ())
    object.__setattr__(block, "statements", (block,))
    program = wrap(pair.tensors, (pair.x,), (pair.y,), (block, pair.copy_loop()))
    expect_defect(program, "cyclic_structure")


def test_forged_cyclic_expression_rejected():
    pair = _VecPair()
    expr = BinaryExpr(
        nid(), BinaryOp.ADD, FloatConst(nid(), 1.0), FloatConst(nid(), 1.0)
    )
    object.__setattr__(expr, "lhs", expr)
    stmt = Store(nid(), pair.y, (IntConst(nid(), 0),), expr)
    expect_defect(pair.program((stmt,)), "cyclic_structure")


def test_list_valued_block_statements_rejected():
    pair = _VecPair()
    block = Block(nid(), ())
    object.__setattr__(block, "statements", [])
    program = wrap(pair.tensors, (pair.x,), (pair.y,), (block, pair.copy_loop()))
    expect_defect(program, "malformed_state")


def test_list_valued_load_indices_rejected():
    pair = _VecPair()
    load = Load(nid(), pair.x, (IntConst(nid(), 0),))
    object.__setattr__(load, "indices", [IntConst(nid(), 0)])
    stmt = Store(nid(), pair.y, (IntConst(nid(), 0),), load)
    expect_defect(pair.program((stmt,)), "malformed_state")


def test_unregistered_expr_subclass_rejected():
    class MysteryExpr(Expr):
        pass

    pair = _VecPair()
    stmt = Store(nid(), pair.y, (IntConst(nid(), 0),), MysteryExpr(nid()))
    expect_defect(pair.program((stmt,)), "unknown_expr")


def test_unregistered_stmt_subclass_rejected():
    class MysteryStmt(Stmt):
        pass

    pair = _VecPair()
    program = wrap(
        pair.tensors,
        (pair.x,),
        (pair.y,),
        (MysteryStmt(nid()), pair.copy_loop()),
    )
    expect_defect(program, "unknown_stmt")


def test_non_expr_extent_rejected():
    pair = _VecPair()
    i = new_index_id()
    loop = DenseFor(nid(), i, 42, Block(nid(), ()))
    expect_defect(pair.program((loop, pair.copy_loop())), "unknown_expr")


def test_non_block_body_rejected():
    pair = _VecPair()
    i = new_index_id()
    loop = DenseFor(
        nid(),
        i,
        DimSize(nid(), pair.x, 0),
        Store(nid(), pair.y, (IntConst(nid(), 0),), FloatConst(nid(), 1.0)),
    )
    expect_defect(pair.program((loop, pair.copy_loop())), "malformed_state")


def test_bool_int_const_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), True),),
        FloatConst(nid(), 1.0),
    )
    expect_defect(pair.program((stmt,)), "malformed_state")


def test_int_valued_float_const_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        FloatConst(nid(), 1),
    )
    expect_defect(pair.program((stmt,)), "malformed_state")


def test_empty_tensor_name_rejected():
    x, y = new_symbol_id(), new_symbol_id()
    tensors = (
        TensorDecl(nid(), x, "", (DENSE,)),
        TensorDecl(nid(), y, "y", (DENSE,)),
    )
    stmt = Store(nid(), y, (IntConst(nid(), 0),), FloatConst(nid(), 1.0))
    program = wrap(tensors, (x,), (y,), (stmt,))
    expect_defect(program, "malformed_state", "program.tensors[0]")


def test_non_level_kind_level_rejected():
    x, y = new_symbol_id(), new_symbol_id()
    tensors = (
        TensorDecl(nid(), x, "x", ("dense",)),
        TensorDecl(nid(), y, "y", (DENSE,)),
    )
    stmt = Store(nid(), y, (IntConst(nid(), 0),), FloatConst(nid(), 1.0))
    program = wrap(tensors, (x,), (y,), (stmt,))
    expect_defect(program, "malformed_state", "program.tensors[0]")


def test_forged_node_id_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst("forged", 0),),
        FloatConst(nid(), 1.0),
    )
    expect_defect(pair.program((stmt,)), "invalid_node_id")


def test_forged_symbol_id_value_rejected():
    pair = _VecPair()
    forged = new_symbol_id()
    object.__setattr__(forged, "value", "zero")
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        Load(nid(), forged, (IntConst(nid(), 0),)),
    )
    expect_defect(pair.program((stmt,)), "invalid_symbol_id")


def _nested_value(depth):
    expr = FloatConst(nid(), 1.0)
    for _ in range(depth):
        expr = BinaryExpr(nid(), BinaryOp.ADD, expr, FloatConst(nid(), 1.0))
    return expr


def test_excessive_expression_depth_rejected():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        _nested_value(MAX_NESTING_DEPTH + 8),
    )
    expect_defect(pair.program((stmt,)), "excessive_depth")


def test_moderate_expression_depth_accepted():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        _nested_value(MAX_NESTING_DEPTH // 2),
    )
    verify_program(pair.program((stmt,)))


def test_non_program_root_rejected():
    with pytest.raises(LoopIRVerificationError) as excinfo:
        verify_program(Block(nid(), ()))
    assert excinfo.value.defect.code == "malformed_state"
