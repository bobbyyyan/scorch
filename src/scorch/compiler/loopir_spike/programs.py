"""Hand-authored candidate LoopIR programs for the Phase-3.5 spike.

These are the milestone's three feasibility cases, written directly in the
generic schema with no operation-specific nodes:

- CSR SpMV — dense outer loop, one sparse cursor, a scalar reduction, and a
  dense store.  It has exactly one sparse cursor, so it exercises sparse
  position iteration but deliberately does not test sparse-sparse merging.
- CSR + CSR elementwise addition — a two-cursor UNION merge assembling a CSR
  output, with per-cursor identity defaults for one-sided coordinates.
- CSR .* CSR elementwise multiplication — a two-cursor INTERSECTION merge
  that genuinely synchronizes both cursors; the body only runs where both
  carry the candidate coordinate, so no defaults exist.

Every builder allocates fresh stable identities, so independently built
fixtures can coexist in one test process.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    DimSize,
    FloatConst,
    IndexValue,
    LevelKind,
    Load,
    LoopProgram,
    MergedSparseFor,
    MergeMode,
    ReduceOp,
    SparseCursorDecl,
    SparseFor,
    Store,
    TensorDecl,
    new_cursor_id,
    new_loop_node_id,
)

_CSR = (LevelKind.DENSE, LevelKind.COMPRESSED)
_VECTOR = (LevelKind.DENSE,)


@dataclass(frozen=True)
class SpmvFixture:
    """The CSR SpMV program plus the symbols a caller binds."""

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


def build_csr_spmv_program() -> SpmvFixture:
    """y[i] = sum_j A[i, j] * x[j] over CSR ``A`` and dense ``x``."""

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
        outer_indices=(IndexValue(new_loop_node_id(), row),),
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
        tensors=(
            TensorDecl(new_loop_node_id(), matrix, "A", _CSR),
            TensorDecl(new_loop_node_id(), vector, "x", _VECTOR),
            TensorDecl(new_loop_node_id(), result, "y", _VECTOR),
        ),
        inputs=(matrix, vector),
        outputs=(result,),
        body=Block(
            new_loop_node_id(),
            (
                DenseFor(
                    new_loop_node_id(),
                    row,
                    DimSize(new_loop_node_id(), matrix, 0),
                    body,
                ),
            ),
        ),
    )
    return SpmvFixture(program, matrix, vector, result)


def _build_elementwise_program(
    mode: MergeMode, op: BinaryOp, with_defaults: bool
) -> ElementwiseFixture:
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

    merge = MergedSparseFor(
        new_loop_node_id(),
        mode,
        (
            SparseCursorDecl(
                node_id=new_loop_node_id(),
                cursor=lhs_cursor,
                tensor=lhs,
                level=1,
                outer_indices=(IndexValue(new_loop_node_id(), row),),
            ),
            SparseCursorDecl(
                node_id=new_loop_node_id(),
                cursor=rhs_cursor,
                tensor=rhs,
                level=1,
                outer_indices=(IndexValue(new_loop_node_id(), row),),
            ),
        ),
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
    program = LoopProgram(
        node_id=new_loop_node_id(),
        tensors=(
            TensorDecl(new_loop_node_id(), lhs, "A", _CSR),
            TensorDecl(new_loop_node_id(), rhs, "B", _CSR),
            TensorDecl(new_loop_node_id(), result, "C", _CSR),
        ),
        inputs=(lhs, rhs),
        outputs=(result,),
        body=Block(
            new_loop_node_id(),
            (
                DenseFor(
                    new_loop_node_id(),
                    row,
                    DimSize(new_loop_node_id(), lhs, 0),
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
