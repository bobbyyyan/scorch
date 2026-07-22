"""Hand-authored candidate LoopIR programs for the Phase-3.5 spike.

The original three feasibility cases plus the repeat-review fixtures, all
written directly in the generic schema with no operation-specific nodes:

- CSR SpMV — dense outer loop, one sparse cursor whose parent is the dense
  level-0 position, a scalar reduction, and a dense store.  It has exactly
  one sparse cursor, so it exercises sparse position iteration but
  deliberately does not test sparse-sparse merging.
- CSR + CSR elementwise addition — a two-cursor UNION merge assembling a CSR
  output, with per-cursor identity defaults for one-sided coordinates.
- CSR .* CSR elementwise multiplication — a two-cursor INTERSECTION merge
  that genuinely synchronizes both cursors; the body only runs where both
  carry the candidate coordinate, so no defaults exist.
- DCSR SpMV — a compressed row level above a compressed column level; the
  inner cursor's parent is the *position* bound by the outer sparse loop,
  proving compressed-under-compressed parent-position descent.
- CSC SpMV — the same logical ``y = A @ x`` with column-major physical
  storage (dense level stores mode 1, compressed level stores mode 0); the
  gathered coordinate is a row, and contributions scatter into the output
  through a reducing store, proving physical/logical mode separation.
- CSF row contraction — a three-level all-compressed tensor computing
  ``y[i] = sum_jk A[i,j,k] * x[k]`` through two chained parent-position
  descents.

Every builder allocates fresh stable identities, so independently built
fixtures can coexist in one test process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..identity import SymbolId, new_index_id, new_symbol_id
from .nodes import (
    Accumulate,
    AccumValue,
    AppendEntry,
    BinaryExpr,
    BinaryOp,
    Block,
    CursorId,
    CursorValue,
    DeclAccum,
    DenseFor,
    DensePosition,
    DimensionDecl,
    Expr,
    FloatConst,
    IndexValue,
    LevelDecl,
    LevelKind,
    Load,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    PositionValue,
    ReduceOp,
    RootPosition,
    SparseCursorDecl,
    SparseFor,
    Store,
    StoreReduce,
    TensorDecl,
    new_cursor_id,
    new_dimension_id,
    new_loop_node_id,
    new_position_id,
)

DENSE = LevelKind.DENSE
COMPRESSED = LevelKind.COMPRESSED


@dataclass(frozen=True)
class SpmvFixture:
    """A matrix-times-vector program plus the symbols a caller binds."""

    program: LoopProgram
    matrix: SymbolId
    vector: SymbolId
    result: SymbolId


@dataclass(frozen=True)
class ElementwiseFixture:
    """A two-CSR elementwise program plus the symbols a caller binds."""

    program: LoopProgram
    lhs: SymbolId
    rhs: SymbolId
    result: SymbolId


def _levels(*pairs: Tuple[LevelKind, int]) -> Tuple[LevelDecl, ...]:
    return tuple(LevelDecl(new_loop_node_id(), kind, mode) for kind, mode in pairs)


def _dense_row_parent(matrix: SymbolId, row_index: Expr) -> DensePosition:
    """The level-0 dense position selected by one bound outer coordinate."""

    return DensePosition(
        new_loop_node_id(),
        matrix,
        0,
        RootPosition(new_loop_node_id()),
        row_index,
    )


def build_csr_spmv_program() -> SpmvFixture:
    """y[i] = sum_j A[i, j] * x[j] over CSR ``A`` and dense ``x``."""

    dim_i = new_dimension_id()
    dim_j = new_dimension_id()
    matrix = new_symbol_id()
    vector = new_symbol_id()
    result = new_symbol_id()
    accumulator = new_symbol_id()
    row = new_index_id()
    column = new_index_id()
    cursor = new_cursor_id()
    cursor_decl = SparseCursorDecl(
        node_id=new_loop_node_id(),
        cursor=cursor,
        tensor=matrix,
        level=1,
        parent=_dense_row_parent(matrix, IndexValue(new_loop_node_id(), row)),
    )
    body = Block(
        new_loop_node_id(),
        (
            DeclAccum(
                new_loop_node_id(),
                accumulator,
                ReduceOp.ADD,
                FloatConst(new_loop_node_id(), 0.0),
            ),
            SparseFor(
                new_loop_node_id(),
                cursor_decl,
                new_position_id(),
                column,
                Block(
                    new_loop_node_id(),
                    (
                        Accumulate(
                            new_loop_node_id(),
                            accumulator,
                            BinaryExpr(
                                new_loop_node_id(),
                                BinaryOp.MUL,
                                CursorValue(new_loop_node_id(), cursor, None),
                                Load(
                                    new_loop_node_id(),
                                    vector,
                                    (IndexValue(new_loop_node_id(), column),),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            Store(
                new_loop_node_id(),
                result,
                (IndexValue(new_loop_node_id(), row),),
                AccumValue(new_loop_node_id(), accumulator),
            ),
        ),
    )
    program = LoopProgram(
        node_id=new_loop_node_id(),
        dimensions=(
            DimensionDecl(new_loop_node_id(), dim_i, "i"),
            DimensionDecl(new_loop_node_id(), dim_j, "j"),
        ),
        tensors=(
            TensorDecl(
                new_loop_node_id(),
                matrix,
                "A",
                (dim_i, dim_j),
                _levels((DENSE, 0), (COMPRESSED, 1)),
            ),
            TensorDecl(new_loop_node_id(), vector, "x", (dim_j,), _levels((DENSE, 0))),
            TensorDecl(new_loop_node_id(), result, "y", (dim_i,), _levels((DENSE, 0))),
        ),
        inputs=(matrix, vector),
        outputs=(result,),
        body=Block(
            new_loop_node_id(),
            (DenseFor(new_loop_node_id(), row, dim_i, body),),
        ),
    )
    return SpmvFixture(program, matrix, vector, result)


def _build_elementwise_program(
    mode: MergeMode, op: BinaryOp, with_defaults: bool
) -> ElementwiseFixture:
    dim_i = new_dimension_id()
    dim_j = new_dimension_id()
    lhs = new_symbol_id()
    rhs = new_symbol_id()
    result = new_symbol_id()
    row = new_index_id()
    column = new_index_id()
    lhs_cursor = new_cursor_id()
    rhs_cursor = new_cursor_id()

    def cursor_read(cursor_ref: CursorId) -> CursorValue:
        default = FloatConst(new_loop_node_id(), 0.0) if with_defaults else None
        return CursorValue(new_loop_node_id(), cursor_ref, default)

    def leaf_cursor(tensor: SymbolId, cursor_ref: CursorId) -> SparseCursorDecl:
        return SparseCursorDecl(
            node_id=new_loop_node_id(),
            cursor=cursor_ref,
            tensor=tensor,
            level=1,
            parent=_dense_row_parent(tensor, IndexValue(new_loop_node_id(), row)),
        )

    merge = MergedSparseFor(
        new_loop_node_id(),
        mode,
        (leaf_cursor(lhs, lhs_cursor), leaf_cursor(rhs, rhs_cursor)),
        column,
        Block(
            new_loop_node_id(),
            (
                AppendEntry(
                    new_loop_node_id(),
                    result,
                    (
                        IndexValue(new_loop_node_id(), row),
                        IndexValue(new_loop_node_id(), column),
                    ),
                    BinaryExpr(
                        new_loop_node_id(),
                        op,
                        cursor_read(lhs_cursor),
                        cursor_read(rhs_cursor),
                    ),
                ),
            ),
        ),
    )
    csr = ((DENSE, 0), (COMPRESSED, 1))
    program = LoopProgram(
        node_id=new_loop_node_id(),
        dimensions=(
            DimensionDecl(new_loop_node_id(), dim_i, "i"),
            DimensionDecl(new_loop_node_id(), dim_j, "j"),
        ),
        tensors=(
            TensorDecl(new_loop_node_id(), lhs, "A", (dim_i, dim_j), _levels(*csr)),
            TensorDecl(new_loop_node_id(), rhs, "B", (dim_i, dim_j), _levels(*csr)),
            TensorDecl(new_loop_node_id(), result, "C", (dim_i, dim_j), _levels(*csr)),
        ),
        inputs=(lhs, rhs),
        outputs=(result,),
        body=Block(
            new_loop_node_id(),
            (
                DenseFor(
                    new_loop_node_id(),
                    row,
                    dim_i,
                    Block(new_loop_node_id(), (merge,)),
                ),
            ),
        ),
    )
    return ElementwiseFixture(program, lhs, rhs, result)


def build_csr_union_add_program() -> ElementwiseFixture:
    """C = A + B over two CSR operands via a UNION merge."""

    return _build_elementwise_program(MergeMode.UNION, BinaryOp.ADD, True)


def build_csr_intersection_multiply_program() -> ElementwiseFixture:
    """C = A .* B over two CSR operands via an INTERSECTION merge."""

    return _build_elementwise_program(MergeMode.INTERSECTION, BinaryOp.MUL, False)


def build_dcsr_spmv_program() -> SpmvFixture:
    """y[i] = sum_j A[i, j] * x[j] over doubly compressed ``A``.

    The outer sparse loop binds the level-0 storage position; the inner
    cursor names that bound position as its dominating parent, which is the
    compressed-under-compressed descent the superseded schema could not
    represent.
    """

    dim_i = new_dimension_id()
    dim_j = new_dimension_id()
    matrix = new_symbol_id()
    vector = new_symbol_id()
    result = new_symbol_id()
    accumulator = new_symbol_id()
    row = new_index_id()
    column = new_index_id()
    row_cursor = new_cursor_id()
    column_cursor = new_cursor_id()
    row_position = new_position_id()
    outer_decl = SparseCursorDecl(
        node_id=new_loop_node_id(),
        cursor=row_cursor,
        tensor=matrix,
        level=0,
        parent=RootPosition(new_loop_node_id()),
    )
    inner_decl = SparseCursorDecl(
        node_id=new_loop_node_id(),
        cursor=column_cursor,
        tensor=matrix,
        level=1,
        parent=PositionValue(new_loop_node_id(), row_position),
    )
    inner_loop = SparseFor(
        new_loop_node_id(),
        inner_decl,
        new_position_id(),
        column,
        Block(
            new_loop_node_id(),
            (
                Accumulate(
                    new_loop_node_id(),
                    accumulator,
                    BinaryExpr(
                        new_loop_node_id(),
                        BinaryOp.MUL,
                        CursorValue(new_loop_node_id(), column_cursor, None),
                        Load(
                            new_loop_node_id(),
                            vector,
                            (IndexValue(new_loop_node_id(), column),),
                        ),
                    ),
                ),
            ),
        ),
    )
    outer_loop = SparseFor(
        new_loop_node_id(),
        outer_decl,
        row_position,
        row,
        Block(
            new_loop_node_id(),
            (
                DeclAccum(
                    new_loop_node_id(),
                    accumulator,
                    ReduceOp.ADD,
                    FloatConst(new_loop_node_id(), 0.0),
                ),
                inner_loop,
                Store(
                    new_loop_node_id(),
                    result,
                    (IndexValue(new_loop_node_id(), row),),
                    AccumValue(new_loop_node_id(), accumulator),
                ),
            ),
        ),
    )
    program = LoopProgram(
        node_id=new_loop_node_id(),
        dimensions=(
            DimensionDecl(new_loop_node_id(), dim_i, "i"),
            DimensionDecl(new_loop_node_id(), dim_j, "j"),
        ),
        tensors=(
            TensorDecl(
                new_loop_node_id(),
                matrix,
                "A",
                (dim_i, dim_j),
                _levels((COMPRESSED, 0), (COMPRESSED, 1)),
            ),
            TensorDecl(new_loop_node_id(), vector, "x", (dim_j,), _levels((DENSE, 0))),
            TensorDecl(new_loop_node_id(), result, "y", (dim_i,), _levels((DENSE, 0))),
        ),
        inputs=(matrix, vector),
        outputs=(result,),
        body=Block(new_loop_node_id(), (outer_loop,)),
    )
    return SpmvFixture(program, matrix, vector, result)


def build_csc_spmv_program() -> SpmvFixture:
    """y[i] = sum_j A[i, j] * x[j] over column-compressed ``A``.

    Physically the dense level stores logical mode 1 (columns) and the
    compressed level stores logical mode 0 (rows), so the traversal is
    column-major while the computed operation is unchanged; the sparse
    coordinate is a *row*, and per-column contributions scatter into the
    dense output through a reducing store.
    """

    dim_i = new_dimension_id()
    dim_j = new_dimension_id()
    matrix = new_symbol_id()
    vector = new_symbol_id()
    result = new_symbol_id()
    row = new_index_id()
    column = new_index_id()
    cursor = new_cursor_id()
    cursor_decl = SparseCursorDecl(
        node_id=new_loop_node_id(),
        cursor=cursor,
        tensor=matrix,
        level=1,
        parent=_dense_row_parent(matrix, IndexValue(new_loop_node_id(), column)),
    )
    sparse_loop = SparseFor(
        new_loop_node_id(),
        cursor_decl,
        new_position_id(),
        row,
        Block(
            new_loop_node_id(),
            (
                StoreReduce(
                    new_loop_node_id(),
                    result,
                    (IndexValue(new_loop_node_id(), row),),
                    ReduceOp.ADD,
                    BinaryExpr(
                        new_loop_node_id(),
                        BinaryOp.MUL,
                        CursorValue(new_loop_node_id(), cursor, None),
                        Load(
                            new_loop_node_id(),
                            vector,
                            (IndexValue(new_loop_node_id(), column),),
                        ),
                    ),
                ),
            ),
        ),
    )
    program = LoopProgram(
        node_id=new_loop_node_id(),
        dimensions=(
            DimensionDecl(new_loop_node_id(), dim_i, "i"),
            DimensionDecl(new_loop_node_id(), dim_j, "j"),
        ),
        tensors=(
            TensorDecl(
                new_loop_node_id(),
                matrix,
                "A",
                (dim_i, dim_j),
                _levels((DENSE, 1), (COMPRESSED, 0)),
            ),
            TensorDecl(new_loop_node_id(), vector, "x", (dim_j,), _levels((DENSE, 0))),
            TensorDecl(new_loop_node_id(), result, "y", (dim_i,), _levels((DENSE, 0))),
        ),
        inputs=(matrix, vector),
        outputs=(result,),
        body=Block(
            new_loop_node_id(),
            (
                DenseFor(
                    new_loop_node_id(),
                    column,
                    dim_j,
                    Block(new_loop_node_id(), (sparse_loop,)),
                ),
            ),
        ),
    )
    return SpmvFixture(program, matrix, vector, result)


def build_csf_row_contraction_program() -> SpmvFixture:
    """y[i] = sum_jk A[i, j, k] * x[k] over three-level all-compressed ``A``.

    Two chained parent-position descents: the level-1 cursor's parent is the
    level-0 bound position and the level-2 cursor's parent is the level-1
    bound position.
    """

    dim_i = new_dimension_id()
    dim_j = new_dimension_id()
    dim_k = new_dimension_id()
    tensor = new_symbol_id()
    vector = new_symbol_id()
    result = new_symbol_id()
    accumulator = new_symbol_id()
    index_i = new_index_id()
    index_j = new_index_id()
    index_k = new_index_id()
    cursor_i = new_cursor_id()
    cursor_j = new_cursor_id()
    cursor_k = new_cursor_id()
    position_i = new_position_id()
    position_j = new_position_id()
    decl_i = SparseCursorDecl(
        node_id=new_loop_node_id(),
        cursor=cursor_i,
        tensor=tensor,
        level=0,
        parent=RootPosition(new_loop_node_id()),
    )
    decl_j = SparseCursorDecl(
        node_id=new_loop_node_id(),
        cursor=cursor_j,
        tensor=tensor,
        level=1,
        parent=PositionValue(new_loop_node_id(), position_i),
    )
    decl_k = SparseCursorDecl(
        node_id=new_loop_node_id(),
        cursor=cursor_k,
        tensor=tensor,
        level=2,
        parent=PositionValue(new_loop_node_id(), position_j),
    )
    innermost = SparseFor(
        new_loop_node_id(),
        decl_k,
        new_position_id(),
        index_k,
        Block(
            new_loop_node_id(),
            (
                Accumulate(
                    new_loop_node_id(),
                    accumulator,
                    BinaryExpr(
                        new_loop_node_id(),
                        BinaryOp.MUL,
                        CursorValue(new_loop_node_id(), cursor_k, None),
                        Load(
                            new_loop_node_id(),
                            vector,
                            (IndexValue(new_loop_node_id(), index_k),),
                        ),
                    ),
                ),
            ),
        ),
    )
    middle = SparseFor(
        new_loop_node_id(),
        decl_j,
        position_j,
        index_j,
        Block(new_loop_node_id(), (innermost,)),
    )
    outer = SparseFor(
        new_loop_node_id(),
        decl_i,
        position_i,
        index_i,
        Block(
            new_loop_node_id(),
            (
                DeclAccum(
                    new_loop_node_id(),
                    accumulator,
                    ReduceOp.ADD,
                    FloatConst(new_loop_node_id(), 0.0),
                ),
                middle,
                Store(
                    new_loop_node_id(),
                    result,
                    (IndexValue(new_loop_node_id(), index_i),),
                    AccumValue(new_loop_node_id(), accumulator),
                ),
            ),
        ),
    )
    program = LoopProgram(
        node_id=new_loop_node_id(),
        dimensions=(
            DimensionDecl(new_loop_node_id(), dim_i, "i"),
            DimensionDecl(new_loop_node_id(), dim_j, "j"),
            DimensionDecl(new_loop_node_id(), dim_k, "k"),
        ),
        tensors=(
            TensorDecl(
                new_loop_node_id(),
                tensor,
                "A",
                (dim_i, dim_j, dim_k),
                _levels((COMPRESSED, 0), (COMPRESSED, 1), (COMPRESSED, 2)),
            ),
            TensorDecl(new_loop_node_id(), vector, "x", (dim_k,), _levels((DENSE, 0))),
            TensorDecl(new_loop_node_id(), result, "y", (dim_i,), _levels((DENSE, 0))),
        ),
        inputs=(tensor, vector),
        outputs=(result,),
        body=Block(new_loop_node_id(), (outer,)),
    )
    return SpmvFixture(program, tensor, vector, result)
