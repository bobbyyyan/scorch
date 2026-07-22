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
    DensePosition,
    DimensionDecl,
    Expr,
    FloatConst,
    IndexValue,
    IntConst,
    LevelDecl,
    LevelKind,
    Load,
    LoopNodeId,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    PositionValue,
    ReduceOp,
    RootPosition,
    SparseCursorDecl,
    SparseFor,
    Stmt,
    Store,
    StoreReduce,
    TensorDecl,
    new_cursor_id,
    new_dimension_id,
    new_loop_node_id,
    new_position_id,
)
from scorch.compiler.loopir_spike.programs import (
    build_csc_spmv_program,
    build_csf_row_contraction_program,
    build_csr_intersection_multiply_program,
    build_csr_spmv_program,
    build_csr_union_add_program,
    build_dcsr_spmv_program,
)
from scorch.compiler.loopir_spike.verifier import (
    MAX_NESTING_DEPTH,
    LoopIRVerificationError,
    verify_program,
)

DENSE = LevelKind.DENSE
COMPRESSED = LevelKind.COMPRESSED


def nid() -> LoopNodeId:
    return new_loop_node_id()


def lvl(kind, mode):
    return LevelDecl(nid(), kind, mode)


def csr_levels():
    return (lvl(DENSE, 0), lvl(COMPRESSED, 1))


def root():
    return RootPosition(nid())


def wrap(dimensions, tensors, inputs, outputs, stmts) -> LoopProgram:
    return LoopProgram(
        nid(), tuple(dimensions), tuple(tensors), inputs, outputs, Block(nid(), stmts)
    )


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
        self.d = new_dimension_id()
        self.x = new_symbol_id()
        self.y = new_symbol_id()
        self.dimensions = (DimensionDecl(nid(), self.d, "d"),)
        self.tensors = (
            TensorDecl(nid(), self.x, "x", (self.d,), (lvl(DENSE, 0),)),
            TensorDecl(nid(), self.y, "y", (self.d,), (lvl(DENSE, 0),)),
        )

    def copy_loop(self):
        i = new_index_id()
        return DenseFor(
            nid(),
            i,
            self.d,
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
        return wrap(self.dimensions, self.tensors, (self.x,), (self.y,), statements)


class _CsrSetup:
    """One CSR input, one CSR output, and merge/cursor building blocks."""

    def __init__(self, second_input=False):
        self.di = new_dimension_id()
        self.dj = new_dimension_id()
        self.dimensions = (
            DimensionDecl(nid(), self.di, "i"),
            DimensionDecl(nid(), self.dj, "j"),
        )
        self.a = new_symbol_id()
        self.c = new_symbol_id()
        dims = (self.di, self.dj)
        tensors = [
            TensorDecl(nid(), self.a, "A", dims, csr_levels()),
            TensorDecl(nid(), self.c, "C", dims, csr_levels()),
        ]
        inputs = [self.a]
        if second_input:
            self.b = new_symbol_id()
            tensors.append(TensorDecl(nid(), self.b, "B", dims, csr_levels()))
            inputs.append(self.b)
        self.tensors = tuple(tensors)
        self.inputs = tuple(inputs)

    def cursor(self, tensor, row_index):
        return SparseCursorDecl(
            node_id=nid(),
            cursor=new_cursor_id(),
            tensor=tensor,
            level=1,
            parent=DensePosition(
                nid(), tensor, 0, root(), IndexValue(nid(), row_index)
            ),
        )

    def program(self, stmts):
        return wrap(self.dimensions, self.tensors, self.inputs, (self.c,), stmts)


class _DcsrSetup:
    """One doubly compressed input and one dense vector output."""

    def __init__(self):
        self.di = new_dimension_id()
        self.dj = new_dimension_id()
        self.dimensions = (
            DimensionDecl(nid(), self.di, "i"),
            DimensionDecl(nid(), self.dj, "j"),
        )
        self.a = new_symbol_id()
        self.y = new_symbol_id()
        self.tensors = (
            TensorDecl(
                nid(),
                self.a,
                "A",
                (self.di, self.dj),
                (lvl(COMPRESSED, 0), lvl(COMPRESSED, 1)),
            ),
            TensorDecl(nid(), self.y, "y", (self.di,), (lvl(DENSE, 0),)),
        )

    def outer_cursor(self):
        return SparseCursorDecl(
            node_id=nid(),
            cursor=new_cursor_id(),
            tensor=self.a,
            level=0,
            parent=root(),
        )

    def inner_cursor(self, parent):
        return SparseCursorDecl(
            node_id=nid(),
            cursor=new_cursor_id(),
            tensor=self.a,
            level=1,
            parent=parent,
        )

    def program(self, stmts):
        return wrap(self.dimensions, self.tensors, (self.a,), (self.y,), stmts)


def test_hand_authored_fixture_programs_verify():
    verify_program(build_csr_spmv_program().program)
    verify_program(build_csr_union_add_program().program)
    verify_program(build_csr_intersection_multiply_program().program)
    verify_program(build_dcsr_spmv_program().program)
    verify_program(build_csc_spmv_program().program)
    verify_program(build_csf_row_contraction_program().program)


def test_reduce_identity_table_is_read_only():
    with pytest.raises(TypeError):
        REDUCE_IDENTITIES[ReduceOp.ADD] = 1.0  # type: ignore[index]


def test_fresh_ids_are_unique():
    assert len({new_loop_node_id() for _ in range(64)}) == 64
    assert len({new_cursor_id() for _ in range(64)}) == 64
    assert len({new_dimension_id() for _ in range(64)}) == 64
    assert len({new_position_id() for _ in range(64)}) == 64


# ------------------------------------------------------ dimension declarations


def test_duplicate_dimension_rejected():
    pair = _VecPair()
    program = wrap(
        pair.dimensions + (DimensionDecl(nid(), pair.d, "d2"),),
        pair.tensors,
        (pair.x,),
        (pair.y,),
        (pair.copy_loop(),),
    )
    expect_defect(program, "duplicate_dimension", "program.dimensions[1]")


def test_undeclared_tensor_dimension_rejected():
    pair = _VecPair()
    stray = new_dimension_id()
    tensors = (
        pair.tensors[0],
        TensorDecl(nid(), pair.y, "y", (stray,), (lvl(DENSE, 0),)),
    )
    program = wrap(pair.dimensions, tensors, (pair.x,), (pair.y,), (pair.copy_loop(),))
    expect_defect(program, "undefined_dimension", "program.tensors[1].dimensions[0]")


def test_dense_for_over_undeclared_dimension_rejected():
    pair = _VecPair()
    loop = DenseFor(nid(), new_index_id(), new_dimension_id(), Block(nid(), ()))
    expect_defect(
        pair.program((loop, pair.copy_loop())),
        "undefined_dimension",
        "program.body.statements[0].dimension",
    )


def test_forged_dimension_id_rejected():
    pair = _VecPair()
    program = wrap(
        (DimensionDecl(nid(), "d", "d"),) + pair.dimensions,
        pair.tensors,
        (pair.x,),
        (pair.y,),
        (pair.copy_loop(),),
    )
    expect_defect(program, "invalid_dimension_id", "program.dimensions[0]")


def test_missing_dimension_id_value_rejected():
    pair = _VecPair()
    object.__delattr__(pair.d, "value")
    expect_defect(pair.program(), "invalid_dimension_id", "program.dimensions[0]")


def test_empty_dimension_name_rejected():
    dimension = new_dimension_id()
    pair = _VecPair()
    program = wrap(
        pair.dimensions + (DimensionDecl(nid(), dimension, ""),),
        pair.tensors,
        (pair.x,),
        (pair.y,),
        (pair.copy_loop(),),
    )
    expect_defect(program, "malformed_state", "program.dimensions[1]")


def test_non_dimension_decl_entry_rejected():
    pair = _VecPair()
    program = wrap(
        pair.dimensions + (IntConst(nid(), 0),),
        pair.tensors,
        (pair.x,),
        (pair.y,),
        (pair.copy_loop(),),
    )
    expect_defect(program, "malformed_state", "program.dimensions[1]")


def test_list_valued_program_dimensions_rejected():
    pair = _VecPair()
    program = pair.program()
    object.__setattr__(program, "dimensions", list(program.dimensions))
    expect_defect(program, "malformed_state", "program.dimensions")


# ----------------------------------------------------------- level declarations


@pytest.mark.parametrize("kind", (LevelKind.COORDINATE, LevelKind.SINGLETON))
def test_coordinate_and_singleton_levels_fail_closed(kind):
    pair = _VecPair()
    tensors = (
        TensorDecl(nid(), pair.x, "x", (pair.d,), (lvl(kind, 0),)),
        pair.tensors[1],
    )
    stmt = Store(nid(), pair.y, (IntConst(nid(), 0),), FloatConst(nid(), 1.0))
    program = wrap(pair.dimensions, tensors, (pair.x,), (pair.y,), (stmt,))
    defect = expect_defect(
        program, "unsupported_level_kind", "program.tensors[0].levels[0]"
    )
    assert kind.value in defect.message


def test_non_level_decl_level_rejected():
    pair = _VecPair()
    tensors = (
        TensorDecl(nid(), pair.x, "x", (pair.d,), (DENSE,)),
        pair.tensors[1],
    )
    stmt = Store(nid(), pair.y, (IntConst(nid(), 0),), FloatConst(nid(), 1.0))
    program = wrap(pair.dimensions, tensors, (pair.x,), (pair.y,), (stmt,))
    expect_defect(program, "malformed_state", "program.tensors[0].levels[0]")


def test_forged_level_kind_rejected():
    pair = _VecPair()
    tensors = (
        TensorDecl(nid(), pair.x, "x", (pair.d,), (LevelDecl(nid(), "dense", 0),)),
        pair.tensors[1],
    )
    stmt = Store(nid(), pair.y, (IntConst(nid(), 0),), FloatConst(nid(), 1.0))
    program = wrap(pair.dimensions, tensors, (pair.x,), (pair.y,), (stmt,))
    expect_defect(program, "malformed_state", "program.tensors[0].levels[0]")


def test_non_int_level_mode_rejected():
    pair = _VecPair()
    tensors = (
        TensorDecl(nid(), pair.x, "x", (pair.d,), (LevelDecl(nid(), DENSE, 0.0),)),
        pair.tensors[1],
    )
    stmt = Store(nid(), pair.y, (IntConst(nid(), 0),), FloatConst(nid(), 1.0))
    program = wrap(pair.dimensions, tensors, (pair.x,), (pair.y,), (stmt,))
    expect_defect(program, "malformed_state", "program.tensors[0].levels[0]")


def test_out_of_range_level_mode_rejected():
    pair = _VecPair()
    tensors = (
        TensorDecl(nid(), pair.x, "x", (pair.d,), (lvl(DENSE, 1),)),
        pair.tensors[1],
    )
    stmt = Store(nid(), pair.y, (IntConst(nid(), 0),), FloatConst(nid(), 1.0))
    program = wrap(pair.dimensions, tensors, (pair.x,), (pair.y,), (stmt,))
    expect_defect(program, "invalid_mode_order", "program.tensors[0].levels[0]")


def test_duplicate_level_modes_rejected():
    setup = _CsrSetup()
    tensors = (
        TensorDecl(
            nid(),
            setup.a,
            "A",
            (setup.di, setup.dj),
            (lvl(DENSE, 0), lvl(COMPRESSED, 0)),
        ),
        setup.tensors[1],
    )
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    program = wrap(setup.dimensions, tensors, (setup.a,), (setup.c,), (append,))
    expect_defect(program, "invalid_mode_order", "program.tensors[0]")


def test_level_count_must_match_dimension_count():
    setup = _CsrSetup()
    tensors = (
        TensorDecl(nid(), setup.a, "A", (setup.di, setup.dj), (lvl(DENSE, 0),)),
        setup.tensors[1],
    )
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    program = wrap(setup.dimensions, tensors, (setup.a,), (setup.c,), (append,))
    expect_defect(program, "rank_mismatch", "program.tensors[0]")


def test_csr_and_csc_declarations_are_distinct():
    csc = (lvl(DENSE, 1), lvl(COMPRESSED, 0))
    csr = csr_levels()
    assert tuple((level.kind, level.mode) for level in csr) != tuple(
        (level.kind, level.mode) for level in csc
    )


# ---------------------------------------------------------------- unique IDs


def test_duplicate_node_id_rejected():
    pair = _VecPair()
    shared = nid()
    i = new_index_id()
    loop = DenseFor(
        nid(),
        i,
        pair.d,
        Block(
            nid(),
            (
                Store(
                    nid(),
                    pair.y,
                    (IndexValue(shared, i),),
                    Load(nid(), pair.x, (IndexValue(shared, i),)),
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
        pair.d,
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
        parent=DensePosition(nid(), setup.a, 0, root(), IndexValue(nid(), i)),
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
    inner = SparseFor(nid(), second, new_position_id(), j2, body)
    outer = SparseFor(nid(), first, new_position_id(), j1, Block(nid(), (inner,)))
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (outer,)))
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
    inner = DenseFor(nid(), i, pair.d, Block(nid(), ()))
    outer = DenseFor(nid(), i, pair.d, Block(nid(), (inner,)))
    expect_defect(pair.program((outer, pair.copy_loop())), "duplicate_index_binding")


def test_duplicate_position_binding_rejected():
    setup = _DcsrSetup()
    position = new_position_id()
    i1, i2 = new_index_id(), new_index_id()
    inner = SparseFor(nid(), setup.outer_cursor(), position, i2, Block(nid(), ()))
    outer = SparseFor(nid(), setup.outer_cursor(), position, i1, Block(nid(), (inner,)))
    store = Store(nid(), setup.y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    expect_defect(
        setup.program((outer, store)),
        "duplicate_position_binding",
        "program.body.statements[0].body.statements[0]",
    )


def test_redeclared_tensor_symbol_rejected():
    pair = _VecPair()
    tensors = pair.tensors + (
        TensorDecl(nid(), pair.x, "x2", (pair.d,), (lvl(DENSE, 0),)),
    )
    program = wrap(pair.dimensions, tensors, (pair.x,), (pair.y,), (pair.copy_loop(),))
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
    sparse = SparseFor(nid(), cursor, new_position_id(), j, Block(nid(), ()))
    late_read = AppendEntry(
        nid(),
        setup.c,
        (IndexValue(nid(), i), IntConst(nid(), 0)),
        CursorValue(nid(), cursor.cursor, None),
    )
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (sparse, late_read)))
    expect_defect(setup.program((loop,)), "unbound_cursor")


def test_position_read_after_loop_rejected():
    setup = _DcsrSetup()
    position = new_position_id()
    i, j = new_index_id(), new_index_id()
    outer = SparseFor(nid(), setup.outer_cursor(), position, i, Block(nid(), ()))
    late = SparseFor(
        nid(),
        setup.inner_cursor(PositionValue(nid(), position)),
        new_position_id(),
        j,
        Block(nid(), ()),
    )
    store = Store(nid(), setup.y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    expect_defect(
        setup.program((outer, late, store)),
        "unbound_position",
        "program.body.statements[1].cursor.parent",
    )


def test_never_bound_position_rejected():
    setup = _DcsrSetup()
    j = new_index_id()
    sparse = SparseFor(
        nid(),
        setup.inner_cursor(PositionValue(nid(), new_position_id())),
        new_position_id(),
        j,
        Block(nid(), ()),
    )
    store = Store(nid(), setup.y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    expect_defect(setup.program((sparse, store)), "unbound_position")


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


def test_dense_position_on_undeclared_tensor_rejected():
    setup = _CsrSetup()
    i, j = new_index_id(), new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        parent=DensePosition(nid(), new_symbol_id(), 0, root(), IndexValue(nid(), i)),
    )
    sparse = SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ()))
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (sparse,)))
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((loop, append)), "undefined_tensor")


def test_undeclared_input_listing_rejected():
    pair = _VecPair()
    program = wrap(
        pair.dimensions,
        pair.tensors,
        (pair.x, new_symbol_id()),
        (pair.y,),
        (pair.copy_loop(),),
    )
    expect_defect(program, "undefined_tensor", "program.inputs[1]")


def test_cursor_parent_using_own_coordinate_rejected():
    setup = _CsrSetup()
    j = new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        parent=DensePosition(nid(), setup.a, 0, root(), IndexValue(nid(), j)),
    )
    sparse = SparseFor(
        nid(),
        decl,
        new_position_id(),
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
        setup.di,
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
        parent=root(),
    )
    j = new_index_id()
    sparse = SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ()))
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
        parent=root(),
    )
    j = new_index_id()
    loop = DenseFor(
        nid(),
        i,
        setup.di,
        Block(nid(), (SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ())),)),
    )
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((loop, append)), "rank_mismatch")


def test_dense_position_on_compressed_level_rejected():
    setup = _DcsrSetup()
    i, j = new_index_id(), new_index_id()
    decl = setup.inner_cursor(
        DensePosition(nid(), setup.a, 0, root(), IndexValue(nid(), i))
    )
    sparse = SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ()))
    outer = SparseFor(
        nid(), setup.outer_cursor(), new_position_id(), i, Block(nid(), (sparse,))
    )
    store = Store(nid(), setup.y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    expect_defect(setup.program((outer, store)), "layout_mismatch")


def test_dense_position_level_out_of_range_rejected():
    setup = _CsrSetup()
    i, j = new_index_id(), new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        parent=DensePosition(nid(), setup.a, 5, root(), IndexValue(nid(), i)),
    )
    sparse = SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ()))
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (sparse,)))
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((loop, append)), "rank_mismatch")


def test_dense_prefix_compressed_leaf_cursor_is_representable():
    d1, d2, d3 = new_dimension_id(), new_dimension_id(), new_dimension_id()
    dy = new_dimension_id()
    x, y = new_symbol_id(), new_symbol_id()
    i, j, k = new_index_id(), new_index_id(), new_index_id()
    level0 = DensePosition(nid(), x, 0, root(), IndexValue(nid(), i))
    level1 = DensePosition(nid(), x, 1, level0, IndexValue(nid(), j))
    cursor = SparseCursorDecl(
        nid(),
        new_cursor_id(),
        x,
        2,
        level1,
    )
    sparse = SparseFor(nid(), cursor, new_position_id(), k, Block(nid(), ()))
    loops = DenseFor(
        nid(),
        i,
        d1,
        Block(nid(), (DenseFor(nid(), j, d2, Block(nid(), (sparse,))),)),
    )
    store = Store(nid(), y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    program = wrap(
        (
            DimensionDecl(nid(), d1, "i"),
            DimensionDecl(nid(), d2, "j"),
            DimensionDecl(nid(), d3, "k"),
            DimensionDecl(nid(), dy, "m"),
        ),
        (
            TensorDecl(
                nid(),
                x,
                "x",
                (d1, d2, d3),
                (lvl(DENSE, 0), lvl(DENSE, 1), lvl(COMPRESSED, 2)),
            ),
            TensorDecl(nid(), y, "y", (dy,), (lvl(DENSE, 0),)),
        ),
        (x,),
        (y,),
        (loops, store),
    )
    verify_program(program)


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


# ---------------------------------------------------- parent-position linkage


def test_leaf_cursor_with_root_parent_rejected():
    setup = _CsrSetup()
    j = new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        parent=root(),
    )
    sparse = SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ()))
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(
        setup.program((sparse, append)),
        "parent_position_mismatch",
        "program.body.statements[0].cursor.parent",
    )


def test_level_zero_cursor_with_non_root_parent_rejected():
    setup = _DcsrSetup()
    i1, i2 = new_index_id(), new_index_id()
    position = new_position_id()
    inner = SparseFor(
        nid(),
        SparseCursorDecl(
            node_id=nid(),
            cursor=new_cursor_id(),
            tensor=setup.a,
            level=0,
            parent=PositionValue(nid(), position),
        ),
        new_position_id(),
        i2,
        Block(nid(), ()),
    )
    outer = SparseFor(nid(), setup.outer_cursor(), position, i1, Block(nid(), (inner,)))
    store = Store(nid(), setup.y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    expect_defect(setup.program((outer, store)), "parent_position_mismatch")


def test_coordinate_typed_cursor_parent_rejected():
    setup = _CsrSetup()
    i, j = new_index_id(), new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        parent=IndexValue(nid(), i),
    )
    sparse = SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ()))
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (sparse,)))
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((loop, append)), "parent_position_mismatch")


def test_wrong_tensor_parent_position_rejected():
    di, dj = new_dimension_id(), new_dimension_id()
    a, b, y = new_symbol_id(), new_symbol_id(), new_symbol_id()
    dcsr = lambda: (lvl(COMPRESSED, 0), lvl(COMPRESSED, 1))  # noqa: E731
    i, j = new_index_id(), new_index_id()
    a_position = new_position_id()
    outer = SparseFor(
        nid(),
        SparseCursorDecl(nid(), new_cursor_id(), a, 0, root()),
        a_position,
        i,
        Block(
            nid(),
            (
                SparseFor(
                    nid(),
                    SparseCursorDecl(
                        nid(),
                        new_cursor_id(),
                        b,
                        1,
                        PositionValue(nid(), a_position),
                    ),
                    new_position_id(),
                    j,
                    Block(nid(), ()),
                ),
            ),
        ),
    )
    store = Store(nid(), y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    program = wrap(
        (DimensionDecl(nid(), di, "i"), DimensionDecl(nid(), dj, "j")),
        (
            TensorDecl(nid(), a, "A", (di, dj), dcsr()),
            TensorDecl(nid(), b, "B", (di, dj), dcsr()),
            TensorDecl(nid(), y, "y", (di,), (lvl(DENSE, 0),)),
        ),
        (a, b),
        (y,),
        (outer, store),
    )
    expect_defect(program, "parent_position_mismatch")


def test_grandparent_position_rejected_for_leaf_descent():
    dims = tuple(new_dimension_id() for _ in range(3))
    a, y = new_symbol_id(), new_symbol_id()
    i, j, k = new_index_id(), new_index_id(), new_index_id()
    position_i = new_position_id()
    leaf = SparseFor(
        nid(),
        SparseCursorDecl(
            nid(), new_cursor_id(), a, 2, PositionValue(nid(), position_i)
        ),
        new_position_id(),
        k,
        Block(nid(), ()),
    )
    middle = SparseFor(
        nid(),
        SparseCursorDecl(
            nid(), new_cursor_id(), a, 1, PositionValue(nid(), position_i)
        ),
        new_position_id(),
        j,
        Block(nid(), (leaf,)),
    )
    outer = SparseFor(
        nid(),
        SparseCursorDecl(nid(), new_cursor_id(), a, 0, root()),
        position_i,
        i,
        Block(nid(), (middle,)),
    )
    store = Store(nid(), y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    program = wrap(
        tuple(
            DimensionDecl(nid(), dim, name) for dim, name in zip(dims, ("i", "j", "k"))
        ),
        (
            TensorDecl(
                nid(),
                a,
                "A",
                dims,
                (lvl(COMPRESSED, 0), lvl(COMPRESSED, 1), lvl(COMPRESSED, 2)),
            ),
            TensorDecl(nid(), y, "y", (dims[0],), (lvl(DENSE, 0),)),
        ),
        (a,),
        (y,),
        (outer, store),
    )
    expect_defect(program, "parent_position_mismatch")


def test_dense_position_parent_linkage_is_checked():
    d1, d2, d3 = new_dimension_id(), new_dimension_id(), new_dimension_id()
    x, y = new_symbol_id(), new_symbol_id()
    i, j, k = new_index_id(), new_index_id(), new_index_id()
    skipped_parent = DensePosition(nid(), x, 1, root(), IndexValue(nid(), j))
    cursor = SparseCursorDecl(nid(), new_cursor_id(), x, 2, skipped_parent)
    sparse = SparseFor(nid(), cursor, new_position_id(), k, Block(nid(), ()))
    loops = DenseFor(
        nid(),
        i,
        d1,
        Block(nid(), (DenseFor(nid(), j, d2, Block(nid(), (sparse,))),)),
    )
    store = Store(nid(), y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    program = wrap(
        (
            DimensionDecl(nid(), d1, "i"),
            DimensionDecl(nid(), d2, "j"),
            DimensionDecl(nid(), d3, "k"),
        ),
        (
            TensorDecl(
                nid(),
                x,
                "x",
                (d1, d2, d3),
                (lvl(DENSE, 0), lvl(DENSE, 1), lvl(COMPRESSED, 2)),
            ),
            TensorDecl(nid(), y, "y", (d1,), (lvl(DENSE, 0),)),
        ),
        (x,),
        (y,),
        (loops, store),
    )
    expect_defect(program, "parent_position_mismatch")


def test_position_value_in_coordinate_slot_rejected():
    setup = _DcsrSetup()
    i = new_index_id()
    position = new_position_id()
    body = Block(
        nid(),
        (
            Store(
                nid(),
                setup.y,
                (PositionValue(nid(), position),),
                FloatConst(nid(), 1.0),
            ),
        ),
    )
    outer = SparseFor(nid(), setup.outer_cursor(), position, i, body)
    expect_defect(setup.program((outer,)), "type_mismatch")


def test_root_position_in_value_slot_rejected():
    pair = _VecPair()
    accumulator = new_symbol_id()
    stmts = (
        DeclAccum(nid(), accumulator, ReduceOp.ADD, FloatConst(nid(), 0.0)),
        Accumulate(nid(), accumulator, root()),
        pair.copy_loop(),
    )
    expect_defect(pair.program(stmts), "type_mismatch")


def test_forged_position_id_rejected():
    setup = _DcsrSetup()
    i = new_index_id()
    outer = SparseFor(nid(), setup.outer_cursor(), "forged", i, Block(nid(), ()))
    store = Store(nid(), setup.y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    expect_defect(setup.program((outer, store)), "invalid_position_id")


def test_missing_position_id_value_rejected():
    setup = _DcsrSetup()
    i = new_index_id()
    position = new_position_id()
    object.__delattr__(position, "value")
    outer = SparseFor(nid(), setup.outer_cursor(), position, i, Block(nid(), ()))
    store = Store(nid(), setup.y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    expect_defect(setup.program((outer, store)), "invalid_position_id")


# ---------------------------------------------------------- value ownership


def test_non_leaf_cursor_value_rejected():
    setup = _DcsrSetup()
    i = new_index_id()
    cursor = setup.outer_cursor()
    body = Block(
        nid(),
        (
            Store(
                nid(),
                setup.y,
                (IndexValue(nid(), i),),
                CursorValue(nid(), cursor.cursor, None),
            ),
        ),
    )
    outer = SparseFor(nid(), cursor, new_position_id(), i, body)
    expect_defect(setup.program((outer,)), "non_leaf_value")


def test_leaf_cursor_value_is_accepted():
    verify_program(build_dcsr_spmv_program().program)


# ------------------------------------------------------------------- typing


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


def test_value_typed_dense_position_coordinate_rejected():
    setup = _CsrSetup()
    j = new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        parent=DensePosition(nid(), setup.a, 0, root(), FloatConst(nid(), 0.0)),
    )
    sparse = SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ()))
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((sparse, append)), "type_mismatch")


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
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (merge,)))
    expect_defect(setup.program((loop,)), "type_mismatch")


# ------------------------------------------------------------------- domains


def test_load_index_from_wrong_dimension_rejected():
    setup = _CsrSetup()
    vec = new_symbol_id()
    tensors = setup.tensors + (
        TensorDecl(nid(), vec, "x", (setup.dj,), (lvl(DENSE, 0),)),
    )
    i = new_index_id()
    loop = DenseFor(
        nid(),
        i,
        setup.di,
        Block(
            nid(),
            (
                AppendEntry(
                    nid(),
                    setup.c,
                    (IndexValue(nid(), i), IntConst(nid(), 0)),
                    Load(nid(), vec, (IndexValue(nid(), i),)),
                ),
            ),
        ),
    )
    program = wrap(
        setup.dimensions,
        tensors,
        setup.inputs + (vec,),
        (setup.c,),
        (loop,),
    )
    defect = expect_defect(program, "domain_mismatch")
    assert "'i'" in defect.message and "'j'" in defect.message


def test_store_index_from_wrong_dimension_rejected():
    setup = _CsrSetup()
    y = new_symbol_id()
    tensors = (
        setup.tensors[0],
        TensorDecl(nid(), y, "y", (setup.di,), (lvl(DENSE, 0),)),
    )
    i, j = new_index_id(), new_index_id()
    cursor = setup.cursor(setup.a, i)
    sparse = SparseFor(
        nid(),
        cursor,
        new_position_id(),
        j,
        Block(
            nid(),
            (
                Store(
                    nid(),
                    y,
                    (IndexValue(nid(), j),),
                    CursorValue(nid(), cursor.cursor, None),
                ),
            ),
        ),
    )
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (sparse,)))
    program = wrap(setup.dimensions, tensors, setup.inputs, (y,), (loop,))
    expect_defect(program, "domain_mismatch")


def test_swapped_append_coordinates_rejected():
    setup = _CsrSetup()
    i, j = new_index_id(), new_index_id()
    cursor = setup.cursor(setup.a, i)
    sparse = SparseFor(
        nid(),
        cursor,
        new_position_id(),
        j,
        Block(
            nid(),
            (
                AppendEntry(
                    nid(),
                    setup.c,
                    (IndexValue(nid(), j), IndexValue(nid(), i)),
                    CursorValue(nid(), cursor.cursor, None),
                ),
            ),
        ),
    )
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (sparse,)))
    expect_defect(setup.program((loop,)), "domain_mismatch")


def test_dense_position_coordinate_from_wrong_dimension_rejected():
    setup = _CsrSetup()
    j = new_index_id()
    outer_j = new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        parent=DensePosition(nid(), setup.a, 0, root(), IndexValue(nid(), outer_j)),
    )
    sparse = SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ()))
    loop = DenseFor(nid(), outer_j, setup.dj, Block(nid(), (sparse,)))
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((loop, append)), "domain_mismatch")


def test_domain_free_constant_coordinates_are_accepted():
    setup = _CsrSetup()
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    verify_program(setup.program((append,)))


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


# --------------------------------------------------------------- StoreReduce


def test_store_reduce_verifies_on_dense_output():
    pair = _VecPair()
    stmt = StoreReduce(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        ReduceOp.ADD,
        FloatConst(nid(), 2.0),
    )
    verify_program(pair.program((stmt,)))


def test_store_reduce_forged_op_rejected():
    pair = _VecPair()
    stmt = StoreReduce(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        "add",
        FloatConst(nid(), 2.0),
    )
    expect_defect(pair.program((stmt,)), "malformed_state")


def test_store_reduce_to_input_rejected():
    pair = _VecPair()
    stmt = StoreReduce(
        nid(),
        pair.x,
        (IntConst(nid(), 0),),
        ReduceOp.ADD,
        FloatConst(nid(), 2.0),
    )
    expect_defect(pair.program((stmt, pair.copy_loop())), "output_scope")


def test_store_reduce_to_compressed_output_rejected():
    setup = _CsrSetup()
    stmt = StoreReduce(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        ReduceOp.ADD,
        FloatConst(nid(), 2.0),
    )
    expect_defect(setup.program((stmt,)), "layout_mismatch")


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
        parent=DensePosition(nid(), setup.c, 0, root(), IndexValue(nid(), i)),
    )
    j = new_index_id()
    loop = DenseFor(
        nid(),
        i,
        setup.di,
        Block(nid(), (SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ())),)),
    )
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((loop, append)), "output_read")


def test_dense_position_on_output_rejected():
    setup = _CsrSetup()
    i, j = new_index_id(), new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        parent=DensePosition(nid(), setup.c, 0, root(), IndexValue(nid(), i)),
    )
    sparse = SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ()))
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (sparse,)))
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((loop, append)), "output_read")


def test_unwritten_output_rejected():
    pair = _VecPair()
    program = wrap(pair.dimensions, pair.tensors, (pair.x,), (pair.y,), ())
    expect_defect(program, "unwritten_output", "program.outputs")


def test_roleless_tensor_rejected():
    pair = _VecPair()
    tensors = pair.tensors + (
        TensorDecl(nid(), new_symbol_id(), "orphan", (pair.d,), (lvl(DENSE, 0),)),
    )
    program = wrap(pair.dimensions, tensors, (pair.x,), (pair.y,), (pair.copy_loop(),))
    expect_defect(program, "output_scope", "program.tensors")


def test_program_without_outputs_rejected():
    pair = _VecPair()
    program = wrap(
        pair.dimensions, pair.tensors, (pair.x, pair.y), (), (pair.copy_loop(),)
    )
    expect_defect(program, "output_scope", "program.outputs")


def test_tensor_in_both_roles_rejected():
    pair = _VecPair()
    program = wrap(
        pair.dimensions,
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
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (merge,)))
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
        new_position_id(),
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
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (sparse,)))
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


def test_merged_cursors_from_different_dimensions_rejected():
    setup = _CsrSetup()
    transposed = new_symbol_id()
    tensors = setup.tensors + (
        TensorDecl(
            nid(),
            transposed,
            "B",
            (setup.di, setup.dj),
            (lvl(DENSE, 1), lvl(COMPRESSED, 0)),
        ),
    )
    i, jj, j = new_index_id(), new_index_id(), new_index_id()
    left = setup.cursor(setup.a, i)
    right = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=transposed,
        level=1,
        parent=DensePosition(nid(), transposed, 0, root(), IndexValue(nid(), jj)),
    )
    merge = MergedSparseFor(
        nid(),
        MergeMode.INTERSECTION,
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
                        BinaryOp.MUL,
                        CursorValue(nid(), left.cursor, None),
                        CursorValue(nid(), right.cursor, None),
                    ),
                ),
            ),
        ),
    )
    inner = DenseFor(nid(), jj, setup.dj, Block(nid(), (merge,)))
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (inner,)))
    program = wrap(
        setup.dimensions,
        tensors,
        setup.inputs + (transposed,),
        (setup.c,),
        (loop,),
    )
    defect = expect_defect(
        program,
        "merge_domain_mismatch",
        "program.body.statements[0].body.statements[0].body.statements[0]",
    )
    assert "'i'" in defect.message and "'j'" in defect.message


def test_merged_non_leaf_cursors_rejected():
    di, dj = new_dimension_id(), new_dimension_id()
    a, b, y = new_symbol_id(), new_symbol_id(), new_symbol_id()
    dcsr = ((COMPRESSED, 0), (COMPRESSED, 1))
    i = new_index_id()
    merge = MergedSparseFor(
        nid(),
        MergeMode.UNION,
        (
            SparseCursorDecl(nid(), new_cursor_id(), a, 0, root()),
            SparseCursorDecl(nid(), new_cursor_id(), b, 0, root()),
        ),
        i,
        Block(nid(), ()),
    )
    store = Store(nid(), y, (IntConst(nid(), 0),), FloatConst(nid(), 0.0))
    program = wrap(
        (DimensionDecl(nid(), di, "i"), DimensionDecl(nid(), dj, "j")),
        (
            TensorDecl(nid(), a, "A", (di, dj), tuple(lvl(k, m) for k, m in dcsr)),
            TensorDecl(nid(), b, "B", (di, dj), tuple(lvl(k, m) for k, m in dcsr)),
            TensorDecl(nid(), y, "y", (di,), (lvl(DENSE, 0),)),
        ),
        (a, b),
        (y,),
        (merge, store),
    )
    expect_defect(
        program,
        "unsupported_sparse_hierarchy",
        "program.body.statements[0].cursors[0]",
    )


# ------------------------------------- cycles, malformed state, and depth


def test_forged_cyclic_block_rejected():
    pair = _VecPair()
    block = Block(nid(), ())
    object.__setattr__(block, "statements", (block,))
    program = wrap(
        pair.dimensions,
        pair.tensors,
        (pair.x,),
        (pair.y,),
        (block, pair.copy_loop()),
    )
    expect_defect(program, "cyclic_structure")


def test_forged_cyclic_expression_rejected():
    pair = _VecPair()
    expr = BinaryExpr(
        nid(), BinaryOp.ADD, FloatConst(nid(), 1.0), FloatConst(nid(), 1.0)
    )
    object.__setattr__(expr, "lhs", expr)
    stmt = Store(nid(), pair.y, (IntConst(nid(), 0),), expr)
    expect_defect(pair.program((stmt,)), "cyclic_structure")


def test_forged_cyclic_dense_position_rejected():
    setup = _CsrSetup()
    i, j = new_index_id(), new_index_id()
    parent = DensePosition(nid(), setup.a, 0, root(), IndexValue(nid(), i))
    object.__setattr__(parent, "parent", parent)
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        parent=parent,
    )
    sparse = SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ()))
    loop = DenseFor(nid(), i, setup.di, Block(nid(), (sparse,)))
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((loop, append)), "cyclic_structure")


def test_list_valued_block_statements_rejected():
    pair = _VecPair()
    block = Block(nid(), ())
    object.__setattr__(block, "statements", [])
    program = wrap(
        pair.dimensions,
        pair.tensors,
        (pair.x,),
        (pair.y,),
        (block, pair.copy_loop()),
    )
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
        pair.dimensions,
        pair.tensors,
        (pair.x,),
        (pair.y,),
        (MysteryStmt(nid()), pair.copy_loop()),
    )
    expect_defect(program, "unknown_stmt")


def test_non_expr_cursor_parent_rejected():
    setup = _CsrSetup()
    j = new_index_id()
    decl = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=setup.a,
        level=1,
        parent=42,
    )
    sparse = SparseFor(nid(), decl, new_position_id(), j, Block(nid(), ()))
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 1.0),
    )
    expect_defect(setup.program((sparse, append)), "unknown_expr")


def test_non_block_body_rejected():
    pair = _VecPair()
    i = new_index_id()
    loop = DenseFor(
        nid(),
        i,
        pair.d,
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
    pair = _VecPair()
    tensors = (
        TensorDecl(nid(), pair.x, "", (pair.d,), (lvl(DENSE, 0),)),
        pair.tensors[1],
    )
    stmt = Store(nid(), pair.y, (IntConst(nid(), 0),), FloatConst(nid(), 1.0))
    program = wrap(pair.dimensions, tensors, (pair.x,), (pair.y,), (stmt,))
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


def test_missing_node_id_value_rejected_with_stable_defect():
    pair = _VecPair()
    stmt = Store(
        nid(),
        pair.y,
        (IntConst(nid(), 0),),
        FloatConst(nid(), 1.0),
    )
    object.__delattr__(stmt.node_id, "value")
    defect = expect_defect(pair.program((stmt,)), "invalid_node_id")
    assert defect.path == "program.body.statements[0]"


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


def test_missing_symbol_id_value_rejected_with_stable_defect():
    pair = _VecPair()
    object.__delattr__(pair.x, "value")
    defect = expect_defect(pair.program(), "invalid_symbol_id")
    assert defect.path == "program.tensors[0]"


def test_forged_index_id_value_rejected():
    pair = _VecPair()
    loop = pair.copy_loop()
    object.__setattr__(loop.index, "value", "zero")
    defect = expect_defect(pair.program((loop,)), "invalid_index_id")
    assert defect.path == "program.body.statements[0]"


def test_missing_cursor_id_value_rejected():
    setup = _CsrSetup()
    row = new_index_id()
    cursor = setup.cursor(setup.a, row)
    object.__delattr__(cursor.cursor, "value")
    loop = DenseFor(
        nid(),
        row,
        setup.di,
        Block(
            nid(),
            (
                SparseFor(
                    nid(),
                    cursor,
                    new_position_id(),
                    new_index_id(),
                    Block(nid(), ()),
                ),
            ),
        ),
    )
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 0.0),
    )
    defect = expect_defect(setup.program((loop, append)), "invalid_cursor_id")
    assert defect.path == "program.body.statements[0].body.statements[0].cursor"


@pytest.mark.parametrize(
    "field", ("dimensions", "tensors", "inputs", "outputs", "body")
)
def test_missing_program_fields_rejected_at_exact_path(field):
    program = _VecPair().program()
    object.__delattr__(program, field)
    defect = expect_defect(program, "malformed_state")
    assert defect.path == f"program.{field}"


def test_missing_nested_node_field_rejected_at_exact_path():
    pair = _VecPair()
    loop = pair.copy_loop()
    object.__delattr__(loop.body, "statements")
    defect = expect_defect(pair.program((loop,)), "malformed_state")
    assert defect.path == "program.body.statements[0].body.statements"


def test_missing_cursor_parent_field_rejected_at_exact_path():
    setup = _CsrSetup()
    i, j = new_index_id(), new_index_id()
    cursor = setup.cursor(setup.a, i)
    object.__delattr__(cursor, "parent")
    loop = DenseFor(
        nid(),
        i,
        setup.di,
        Block(
            nid(),
            (SparseFor(nid(), cursor, new_position_id(), j, Block(nid(), ())),),
        ),
    )
    append = AppendEntry(
        nid(),
        setup.c,
        (IntConst(nid(), 0), IntConst(nid(), 0)),
        FloatConst(nid(), 0.0),
    )
    defect = expect_defect(setup.program((loop, append)), "malformed_state")
    assert defect.path.endswith(".cursor.parent")


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
