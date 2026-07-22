"""Execution and differential coverage for the Phase-3.5 LoopIR spike.

The hand-authored feasibility programs run through the generic schema and
interpreter over plain-Python containers, and every result is compared
against an independent pure-Python dense reference.  The accumulation order
matches the reference exactly (adding a zero term never changes an IEEE
partial sum), so all comparisons are exact — including the cross-layout
comparisons, where one logical matrix stored CSR, DCSR, and CSC must produce
identical SpMV results.
"""

import math
import random

import pytest

from scorch.compiler.loopir_spike import interp as loopir_interp
from scorch.compiler.identity import new_index_id, new_symbol_id
from scorch.compiler.loopir_spike.csr import CsrFormatError, CsrMatrix
from scorch.compiler.loopir_spike.interp import LoopIRInterpreterError, run_program
from scorch.compiler.loopir_spike.levels import (
    CompressedLevel,
    CsrOutputBuilder,
    DenseLevel,
    LevelStorageError,
    LevelTensorStorage,
    from_csr,
)
from scorch.compiler.loopir_spike.nodes import (
    AppendEntry,
    Block,
    CursorValue,
    DenseFor,
    DensePosition,
    DimensionDecl,
    FloatConst,
    IndexValue,
    IntConst,
    LevelDecl,
    LevelKind,
    LoopProgram,
    PositionLoad,
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
from scorch.compiler.loopir_spike.programs import (
    build_csc_spmv_program,
    build_csf_row_contraction_program,
    build_csr_intersection_multiply_program,
    build_csr_spmv_program,
    build_csr_union_add_program,
    build_dcsr_spmv_program,
    build_mixed_dense_leaf_contraction_program,
    build_sparse_dense_spmv_program,
)

nid = new_loop_node_id

DENSE = LevelKind.DENSE
COMPRESSED = LevelKind.COMPRESSED


def dense_spmv(rows, x):
    return [sum(entry * x[j] for j, entry in enumerate(row)) for row in rows]


def dense_add(a, b):
    return [[p + q for p, q in zip(ra, rb)] for ra, rb in zip(a, b)]


def dense_intersect_mul(a, b):
    return [
        [p * q if p != 0.0 and q != 0.0 else 0.0 for p, q in zip(ra, rb)]
        for ra, rb in zip(a, b)
    ]


def random_dense(rng, n_rows, n_cols, density):
    return [
        [
            rng.uniform(-4.0, 4.0) if rng.random() < density else 0.0
            for _ in range(n_cols)
        ]
        for _ in range(n_rows)
    ]


def matrix_storage(dense_rows, n_cols, modes, kinds):
    return LevelTensorStorage.from_dense(
        dense_rows, (len(dense_rows), n_cols), modes, kinds
    )


def run_spmv(dense_rows, n_cols, x):
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense(dense_rows, n_cols)
    results = run_program(
        fixture.program,
        {fixture.matrix: matrix, fixture.vector: x},
        {fixture.result: (matrix.n_rows,)},
    )
    return results[fixture.result]


def run_dcsr_spmv(dense_rows, n_cols, x):
    fixture = build_dcsr_spmv_program()
    storage = matrix_storage(dense_rows, n_cols, (0, 1), (COMPRESSED, COMPRESSED))
    results = run_program(
        fixture.program,
        {fixture.matrix: storage, fixture.vector: x},
        {fixture.result: (len(dense_rows),)},
    )
    return results[fixture.result]


def run_csc_spmv(dense_rows, n_cols, x):
    fixture = build_csc_spmv_program()
    storage = matrix_storage(dense_rows, n_cols, (1, 0), (DENSE, COMPRESSED))
    results = run_program(
        fixture.program,
        {fixture.matrix: storage, fixture.vector: x},
        {fixture.result: (len(dense_rows),)},
    )
    return results[fixture.result]


def run_csf_row_contraction(cube, shape, x):
    fixture = build_csf_row_contraction_program()
    storage = LevelTensorStorage.from_dense(
        cube, shape, (0, 1, 2), (COMPRESSED, COMPRESSED, COMPRESSED)
    )
    results = run_program(
        fixture.program,
        {fixture.matrix: storage, fixture.vector: x},
        {fixture.result: (shape[0],)},
    )
    return results[fixture.result]


def run_sparse_dense_spmv(dense_rows, n_cols, x):
    fixture = build_sparse_dense_spmv_program()
    storage = matrix_storage(dense_rows, n_cols, (1, 0), (COMPRESSED, DENSE))
    results = run_program(
        fixture.program,
        {fixture.matrix: storage, fixture.vector: x},
        {fixture.result: (len(dense_rows),)},
    )
    return results[fixture.result]


def run_mixed_dense_leaf_contraction(cube, shape, x):
    fixture = build_mixed_dense_leaf_contraction_program()
    storage = LevelTensorStorage.from_dense(
        cube, shape, (0, 1, 2), (DENSE, COMPRESSED, DENSE)
    )
    results = run_program(
        fixture.program,
        {fixture.matrix: storage, fixture.vector: x},
        {fixture.result: (shape[0],)},
    )
    return results[fixture.result]


def run_elementwise(build, dense_a, dense_b, n_cols):
    fixture = build()
    a = CsrMatrix.from_dense(dense_a, n_cols)
    b = CsrMatrix.from_dense(dense_b, n_cols)
    results = run_program(
        fixture.program,
        {fixture.lhs: a, fixture.rhs: b},
        {fixture.result: (a.n_rows, n_cols)},
    )
    out = results[fixture.result]
    assert type(out) is CsrMatrix
    return out


# --------------------------------------------------------------- CSR SpMV


def test_spmv_small_dense_reference():
    rows = [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]
    x = [1.0, 2.0, 3.0]
    assert run_spmv(rows, 3, x) == dense_spmv(rows, x)


def test_spmv_all_zero_matrix():
    rows = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    assert run_spmv(rows, 2, [5.0, -1.0]) == [0.0, 0.0, 0.0]


def test_spmv_empty_rows_and_ragged_fill():
    rows = [
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 7.0],
        [5.0, 0.0, 6.0, 0.0],
    ]
    x = [2.0, -1.0, 0.5, 1.25]
    assert run_spmv(rows, 4, x) == dense_spmv(rows, x)


def test_spmv_zero_row_matrix():
    assert run_spmv([], 3, [1.0, 2.0, 3.0]) == []


def test_spmv_zero_column_matrix():
    assert run_spmv([[], [], []], 0, []) == [0.0, 0.0, 0.0]


def test_spmv_vector_with_zeros():
    rows = [[2.0, 4.0], [8.0, 16.0]]
    x = [0.0, 3.0]
    assert run_spmv(rows, 2, x) == dense_spmv(rows, x)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (8, 8), (16, 5)])
@pytest.mark.parametrize("density", [0.0, 0.1, 0.4, 0.9])
def test_spmv_randomized_against_dense_reference(seed, shape, density):
    rng = random.Random(1000 * seed + 10 * shape[0] + shape[1])
    rows = random_dense(rng, shape[0], shape[1], density)
    x = [rng.uniform(-4.0, 4.0) for _ in range(shape[1])]
    assert run_spmv(rows, shape[1], x) == dense_spmv(rows, x)


def test_spmv_short_vector_fails_closed():
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[0.0, 0.0, 5.0]], 3)
    with pytest.raises(LoopIRInterpreterError, match="dimension extent mismatch"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: [1.0, 2.0]},
            {fixture.result: (1,)},
        )


def test_fixture_program_is_reusable_across_runs():
    fixture = build_csr_spmv_program()
    for rows, x in (
        ([[1.0, 0.0], [0.0, 2.0]], [3.0, 4.0]),
        ([[0.0, 7.0], [0.0, 0.0]], [1.0, -1.0]),
    ):
        matrix = CsrMatrix.from_dense(rows, 2)
        results = run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: x},
            {fixture.result: (matrix.n_rows,)},
        )
        assert results[fixture.result] == dense_spmv(rows, x)


# ------------------------------------------------------- CSR + CSR (UNION)


def test_union_add_disjoint_support():
    a = [[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    b = [[0.0, 3.0, 0.0], [4.0, 0.0, 0.0]]
    out = run_elementwise(build_csr_union_add_program, a, b, 3)
    assert out.to_dense() == dense_add(a, b)
    assert out.nnz == 4


def test_union_add_identical_support():
    a = [[1.0, 2.0], [0.0, 3.0]]
    b = [[10.0, 20.0], [0.0, 30.0]]
    out = run_elementwise(build_csr_union_add_program, a, b, 2)
    assert out.to_dense() == dense_add(a, b)
    assert out.nnz == 3


def test_union_add_overlapping_and_one_sided_rows():
    a = [
        [1.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 6.0, 0.0],
    ]
    b = [
        [0.0, 7.0, 8.0, 0.0],
        [0.0, 0.0, 0.0, 9.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    out = run_elementwise(build_csr_union_add_program, a, b, 4)
    assert out.to_dense() == dense_add(a, b)
    assert out.nnz == 6


def test_union_add_one_sided_early_exhaustion():
    a = [[1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    b = [[0.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0]]
    out = run_elementwise(build_csr_union_add_program, a, b, 7)
    assert out.to_dense() == dense_add(a, b)
    assert out.indices == (0, 1, 4, 5, 6)


def test_union_add_unequal_row_lengths():
    a = [[1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 2.0, 0.0, 0.0, 0.0]]
    b = [[0.0, 0.0, 3.0, 0.0, 0.0], [4.0, 4.0, 4.0, 4.0, 4.0]]
    out = run_elementwise(build_csr_union_add_program, a, b, 5)
    assert out.to_dense() == dense_add(a, b)


def test_union_add_empty_operands():
    zero = [[0.0, 0.0], [0.0, 0.0]]
    some = [[1.0, 0.0], [0.0, 2.0]]
    for a, b in ((zero, some), (some, zero), (zero, zero)):
        out = run_elementwise(build_csr_union_add_program, a, b, 2)
        assert out.to_dense() == dense_add(a, b)


def test_union_add_zero_shape_operands():
    out = run_elementwise(build_csr_union_add_program, [], [], 3)
    assert out.n_rows == 0 and out.nnz == 0
    out = run_elementwise(build_csr_union_add_program, [[], []], [[], []], 0)
    assert (out.n_rows, out.n_cols, out.nnz) == (2, 0, 0)


def test_union_add_cancellation_keeps_explicit_zero():
    a = [[1.5, 0.0]]
    b = [[-1.5, 2.0]]
    out = run_elementwise(build_csr_union_add_program, a, b, 2)
    assert out.to_dense() == [[0.0, 2.0]]
    assert out.nnz == 2
    assert out.values == (0.0, 2.0)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shape", [(1, 1), (4, 6), (9, 9)])
@pytest.mark.parametrize("densities", [(0.1, 0.7), (0.4, 0.4), (0.0, 0.5)])
def test_union_add_randomized_against_dense_reference(seed, shape, densities):
    rng = random.Random(7000 * seed + 100 * shape[0] + shape[1])
    a = random_dense(rng, shape[0], shape[1], densities[0])
    b = random_dense(rng, shape[0], shape[1], densities[1])
    out = run_elementwise(build_csr_union_add_program, a, b, shape[1])
    assert out.to_dense() == dense_add(a, b)


# ---------------------------------------------- CSR .* CSR (INTERSECTION)


def test_intersection_multiply_disjoint_support_is_empty():
    a = [[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]
    b = [[0.0, 3.0, 0.0], [4.0, 0.0, 0.0]]
    out = run_elementwise(build_csr_intersection_multiply_program, a, b, 3)
    assert out.nnz == 0
    assert out.to_dense() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_intersection_multiply_nested_support():
    a = [[1.0, 2.0, 3.0, 4.0]]
    b = [[0.0, 5.0, 0.0, 6.0]]
    out = run_elementwise(build_csr_intersection_multiply_program, a, b, 4)
    assert out.to_dense() == dense_intersect_mul(a, b)
    assert out.indices == (1, 3)


def test_intersection_multiply_one_side_empty():
    a = [[0.0, 0.0, 0.0]]
    b = [[1.0, 2.0, 3.0]]
    out = run_elementwise(build_csr_intersection_multiply_program, a, b, 3)
    assert out.nnz == 0


def test_intersection_multiply_early_exhaustion():
    a = [[1.0, 0.0, 0.0, 0.0, 0.0]]
    b = [[2.0, 0.0, 3.0, 4.0, 5.0]]
    out = run_elementwise(build_csr_intersection_multiply_program, a, b, 5)
    assert out.to_dense() == dense_intersect_mul(a, b)
    assert out.indices == (0,)


def test_intersection_is_structural_not_value_based():
    a = CsrMatrix(
        n_rows=1,
        n_cols=2,
        indptr=(0, 1),
        indices=(1,),
        values=(0.0,),
    )
    b = CsrMatrix.from_dense([[5.0, 7.0]], 2)
    fixture = build_csr_intersection_multiply_program()
    results = run_program(
        fixture.program,
        {fixture.lhs: a, fixture.rhs: b},
        {fixture.result: (1, 2)},
    )
    out = results[fixture.result]
    assert type(out) is CsrMatrix
    assert out.nnz == 1
    assert out.indices == (1,)
    assert out.values == (0.0,)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("shape", [(1, 1), (4, 6), (9, 9)])
@pytest.mark.parametrize("densities", [(0.1, 0.7), (0.5, 0.5), (0.0, 0.6)])
def test_intersection_randomized_against_dense_reference(seed, shape, densities):
    rng = random.Random(9000 * seed + 100 * shape[0] + shape[1])
    a = random_dense(rng, shape[0], shape[1], densities[0])
    b = random_dense(rng, shape[0], shape[1], densities[1])
    out = run_elementwise(build_csr_intersection_multiply_program, a, b, shape[1])
    assert out.to_dense() == dense_intersect_mul(a, b)


# ---------------------------------------------- DCSR SpMV (nested descent)


def test_dcsr_spmv_small_dense_reference():
    rows = [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]
    x = [1.0, 2.0, 3.0]
    assert run_dcsr_spmv(rows, 3, x) == dense_spmv(rows, x)


def test_dcsr_spmv_row_positions_differ_from_row_coordinates():
    # Row 0 is absent, so stored row coordinates (1, 2) live at storage
    # positions (0, 1).  Confusing the coordinate for the position would
    # select the wrong (or no) child segment and corrupt both sums.
    rows = [
        [0.0, 0.0, 0.0],
        [5.0, 6.0, 0.0],
        [0.0, 0.0, 7.0],
    ]
    x = [1.0, 10.0, 100.0]
    assert run_dcsr_spmv(rows, 3, x) == [0.0, 65.0, 700.0]


def test_dcsr_spmv_all_zero_and_zero_shape():
    assert run_dcsr_spmv([[0.0, 0.0], [0.0, 0.0]], 2, [1.0, 2.0]) == [0.0, 0.0]
    assert run_dcsr_spmv([], 3, [1.0, 2.0, 3.0]) == []


def test_dcsr_spmv_matches_csr_fixture():
    rows = [
        [0.0, 2.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, -3.0],
    ]
    x = [0.5, -1.5, 2.0, 8.0]
    assert run_dcsr_spmv(rows, 4, x) == run_spmv(rows, 4, x)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (8, 8)])
@pytest.mark.parametrize("density", [0.0, 0.15, 0.5, 0.9])
def test_dcsr_spmv_randomized_against_dense_and_csr(seed, shape, density):
    rng = random.Random(3000 * seed + 10 * shape[0] + shape[1])
    rows = random_dense(rng, shape[0], shape[1], density)
    x = [rng.uniform(-4.0, 4.0) for _ in range(shape[1])]
    expected = dense_spmv(rows, x)
    assert run_dcsr_spmv(rows, shape[1], x) == expected
    assert run_spmv(rows, shape[1], x) == expected


# --------------------------------------- CSC SpMV (physical/logical split)


def test_csc_spmv_small_dense_reference():
    rows = [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]
    x = [1.0, 2.0, 3.0]
    assert run_csc_spmv(rows, 3, x) == dense_spmv(rows, x)


def test_csc_spmv_is_not_transposed():
    # A deliberately non-symmetric matrix: reading the compressed level's
    # row coordinates as columns (the superseded schema's only option)
    # would compute A^T @ x instead.
    rows = [
        [0.0, 9.0],
        [4.0, 0.0],
    ]
    x = [1.0, 10.0]
    assert run_csc_spmv(rows, 2, x) == [90.0, 4.0]
    transposed = [[0.0, 4.0], [9.0, 0.0]]
    assert dense_spmv(transposed, x) == [40.0, 9.0]


def test_csc_spmv_zero_shapes():
    assert run_csc_spmv([], 3, [1.0, 2.0, 3.0]) == []
    assert run_csc_spmv([[], [], []], 0, []) == [0.0, 0.0, 0.0]


def test_csc_spmv_matches_csr_fixture():
    rows = [
        [0.0, 2.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, -3.0],
    ]
    x = [0.5, -1.5, 2.0, 8.0]
    assert run_csc_spmv(rows, 4, x) == run_spmv(rows, 4, x)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (8, 8)])
@pytest.mark.parametrize("density", [0.0, 0.15, 0.5, 0.9])
def test_csc_spmv_randomized_against_dense_and_csr(seed, shape, density):
    rng = random.Random(5000 * seed + 10 * shape[0] + shape[1])
    rows = random_dense(rng, shape[0], shape[1], density)
    x = [rng.uniform(-4.0, 4.0) for _ in range(shape[1])]
    expected = dense_spmv(rows, x)
    assert run_csc_spmv(rows, shape[1], x) == expected
    assert run_spmv(rows, shape[1], x) == expected


# -------------------- sparse-dense SpMV (permuted DENSE leaf ownership)


def test_sparse_dense_spmv_reads_dense_leaf_positions():
    rows = [[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]
    x = [1.0, 2.0, 3.0]
    assert run_sparse_dense_spmv(rows, 3, x) == dense_spmv(rows, x)


def test_sparse_dense_spmv_column_positions_differ_from_coordinates():
    # Only logical columns 1 and 3 are stored at outer positions 0 and 1.
    # Using the column coordinate as the dense leaf's parent would select a
    # wrong or out-of-range row block.
    rows = [
        [0.0, 5.0, 0.0, 7.0],
        [0.0, 0.0, 0.0, 11.0],
        [0.0, 13.0, 0.0, 0.0],
    ]
    x = [1.0, 10.0, 100.0, 1000.0]
    assert run_sparse_dense_spmv(rows, 4, x) == [7050.0, 11000.0, 130.0]


def test_sparse_dense_spmv_zero_shapes():
    assert run_sparse_dense_spmv([], 3, [1.0, 2.0, 3.0]) == []
    assert run_sparse_dense_spmv([[], []], 0, []) == [0.0, 0.0]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("shape", [(1, 1), (3, 7), (7, 4)])
@pytest.mark.parametrize("density", [0.0, 0.25, 0.8])
def test_sparse_dense_spmv_randomized_against_other_layouts(seed, shape, density):
    rng = random.Random(17000 * seed + 10 * shape[0] + shape[1])
    rows = random_dense(rng, shape[0], shape[1], density)
    x = [rng.uniform(-4.0, 4.0) for _ in range(shape[1])]
    expected = dense_spmv(rows, x)
    assert run_sparse_dense_spmv(rows, shape[1], x) == expected
    assert run_spmv(rows, shape[1], x) == expected


# ------------------------------------ CSF three-level contraction (descent)


def dense_csf_reference(cube, shape, x):
    return [
        sum(
            cube[i][j][k] * x[k]
            for j in range(shape[1])
            for k in range(shape[2])
            if cube[i][j][k] != 0.0
        )
        for i in range(shape[0])
    ]


def test_csf_row_contraction_small():
    cube = [
        [[1.0, 0.0], [0.0, 2.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 3.0], [4.0, 0.0]],
    ]
    x = [10.0, 100.0]
    assert run_csf_row_contraction(cube, (3, 2, 2), x) == [210.0, 0.0, 340.0]


def test_csf_fiber_positions_differ_from_coordinates():
    # Only fibers (0, 1) and (2, 0) are stored, so every stored level-1
    # segment must be selected by the parent's storage position, and every
    # stored level-2 segment by the level-1 position — coordinate reuse
    # would select missing or wrong fibers.
    cube = [
        [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[7.0, 0.0, 11.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
    x = [1.0, 10.0, 100.0]
    assert run_csf_row_contraction(cube, (3, 3, 3), x) == [50.0, 0.0, 1107.0]


def test_csf_zero_tensor():
    cube = [[[0.0], [0.0]], [[0.0], [0.0]]]
    assert run_csf_row_contraction(cube, (2, 2, 1), [3.0]) == [0.0, 0.0]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (4, 4, 4)])
@pytest.mark.parametrize("density", [0.0, 0.2, 0.6])
def test_csf_randomized_against_dense_reference(seed, shape, density):
    rng = random.Random(11000 * seed + 100 * shape[0] + 10 * shape[1] + shape[2])
    cube = [
        [
            [
                rng.uniform(-4.0, 4.0) if rng.random() < density else 0.0
                for _ in range(shape[2])
            ]
            for _ in range(shape[1])
        ]
        for _ in range(shape[0])
    ]
    x = [rng.uniform(-4.0, 4.0) for _ in range(shape[2])]
    assert run_csf_row_contraction(cube, shape, x) == dense_csf_reference(
        cube, shape, x
    )


# -------------------------- multilevel DENSE/COMPRESSED/DENSE contraction


def test_mixed_dense_leaf_contraction_small():
    cube = [
        [[1.0, 0.0], [0.0, 2.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 3.0], [4.0, 0.0]],
    ]
    x = [10.0, 100.0]
    assert run_mixed_dense_leaf_contraction(cube, (3, 2, 2), x) == [
        210.0,
        0.0,
        340.0,
    ]


def test_mixed_dense_leaf_positions_differ_from_sparse_coordinates():
    cube = [
        [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[7.0, 0.0, 11.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
    x = [1.0, 10.0, 100.0]
    assert run_mixed_dense_leaf_contraction(cube, (3, 3, 3), x) == [
        50.0,
        0.0,
        1107.0,
    ]


def test_mixed_dense_leaf_contraction_zero_shapes():
    assert run_mixed_dense_leaf_contraction([], (0, 2, 3), [1.0, 2.0, 3.0]) == []
    assert run_mixed_dense_leaf_contraction([[]], (1, 0, 3), [1.0, 2.0, 3.0]) == [0.0]
    assert run_mixed_dense_leaf_contraction([[[], []]], (1, 2, 0), []) == [0.0]


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 3, 4), (4, 2, 3)])
@pytest.mark.parametrize("density", [0.0, 0.3, 0.8])
def test_mixed_dense_leaf_randomized_against_dense_reference(seed, shape, density):
    rng = random.Random(19000 * seed + 100 * shape[0] + 10 * shape[1] + shape[2])
    cube = [
        [
            [
                rng.uniform(-4.0, 4.0) if rng.random() < density else 0.0
                for _ in range(shape[2])
            ]
            for _ in range(shape[1])
        ]
        for _ in range(shape[0])
    ]
    x = [rng.uniform(-4.0, 4.0) for _ in range(shape[2])]
    assert run_mixed_dense_leaf_contraction(cube, shape, x) == dense_csf_reference(
        cube, shape, x
    )


# ------------------------------------------- rank-1 compressed iteration


def test_rank_one_compressed_cursor_executes():
    d = new_dimension_id()
    dy = new_dimension_id()
    x = new_symbol_id()
    y = new_symbol_id()
    i = new_index_id()
    cursor = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=x,
        level=0,
        parent=RootPosition(nid()),
    )
    program = LoopProgram(
        nid(),
        (DimensionDecl(nid(), d, "d"), DimensionDecl(nid(), dy, "m")),
        (
            TensorDecl(nid(), x, "x", (d,), (LevelDecl(nid(), COMPRESSED, 0),)),
            TensorDecl(nid(), y, "y", (dy,), (LevelDecl(nid(), DENSE, 0),)),
        ),
        (x,),
        (y,),
        Block(
            nid(),
            (
                SparseFor(
                    nid(),
                    cursor,
                    new_position_id(),
                    i,
                    Block(
                        nid(),
                        (
                            StoreReduce(
                                nid(),
                                y,
                                (IntConst(nid(), 0),),
                                ReduceOp.ADD,
                                CursorValue(nid(), cursor.cursor, None),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    storage = LevelTensorStorage.from_dense(
        [0.0, 2.0, 3.0, 0.0, 4.0], (5,), (0,), (COMPRESSED,)
    )
    results = run_program(program, {x: storage}, {y: (1,)})
    assert results[y] == [9.0]


# ------------------------------------------------------------ CSR container


def test_csr_from_dense_round_trip_drops_zeros():
    rows = [[0.0, 1.5, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 3.0]]
    matrix = CsrMatrix.from_dense(rows, 3)
    assert matrix.nnz == 3
    assert matrix.indptr == (0, 1, 1, 3)
    assert matrix.to_dense() == rows


def test_csr_row_segment_bounds():
    matrix = CsrMatrix.from_dense([[1.0], [0.0]], 1)
    assert matrix.row_segment(0) == (0, 1)
    assert matrix.row_segment(1) == (1, 1)
    with pytest.raises(CsrFormatError):
        matrix.row_segment(2)


@pytest.mark.parametrize("row", (True, 0.0, "0"))
def test_csr_row_segment_rejects_non_exact_integer_rows(row):
    matrix = CsrMatrix.from_dense([[1.0]], 1)
    with pytest.raises(CsrFormatError, match="exact int"):
        matrix.row_segment(row)


def test_csr_from_dense_rejects_unrepresentable_values():
    with pytest.raises(CsrFormatError, match="unrepresentable numeric"):
        CsrMatrix.from_dense([[10**10000]], 1)


def test_csr_from_dense_rejects_sequence_subclasses_without_callbacks():
    class HostileList(list):
        def __iter__(self):
            raise RuntimeError("sequence callback ran")

    with pytest.raises(CsrFormatError, match="owned list or tuple"):
        CsrMatrix.from_dense(HostileList(([1.0],)), 1)
    with pytest.raises(CsrFormatError, match="owned list or tuple"):
        CsrMatrix.from_dense([HostileList((1.0,))], 1)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(indptr=(1, 1), indices=(), values=()), "start at zero"),
        (dict(indptr=(0,), indices=(), values=()), "indptr length"),
        (dict(indptr=(0, 1), indices=(), values=()), "terminate"),
        (
            dict(indptr=(0, 2), indices=(1, 0), values=(1.0, 2.0)),
            "strictly increasing",
        ),
        (
            dict(indptr=(0, 2), indices=(1, 1), values=(1.0, 2.0)),
            "strictly increasing",
        ),
        (dict(indptr=(0, 1), indices=(9,), values=(1.0,)), "outside"),
        (dict(indptr=(0, 1), indices=(0,), values=(1,)), "exact float"),
        (dict(indptr=(0, 1), indices=(0.0,), values=(1.0,)), "exact int"),
        (dict(indptr=(0, 1), indices=(0,), values=()), "equal length"),
    ],
)
def test_csr_non_canonical_construction_rejected(kwargs, match):
    with pytest.raises(CsrFormatError, match=match):
        CsrMatrix(n_rows=1, n_cols=2, **kwargs)


def test_csr_decreasing_indptr_rejected():
    with pytest.raises(CsrFormatError, match="nondecreasing"):
        CsrMatrix(
            n_rows=2,
            n_cols=2,
            indptr=(0, 2, 1),
            indices=(0,),
            values=(1.0,),
        )


def test_csr_list_streams_rejected():
    with pytest.raises(CsrFormatError, match="owned tuple"):
        CsrMatrix(n_rows=1, n_cols=1, indptr=[0, 0], indices=(), values=())


def test_csr_negative_dimensions_rejected():
    with pytest.raises(CsrFormatError, match="nonnegative"):
        CsrMatrix(n_rows=-1, n_cols=1, indptr=(0,), indices=(), values=())


def test_csr_from_dense_ragged_rejected():
    with pytest.raises(CsrFormatError, match="columns"):
        CsrMatrix.from_dense([[1.0, 2.0], [3.0]], 2)


def test_csr_explicit_zero_construction_allowed():
    matrix = CsrMatrix(n_rows=1, n_cols=1, indptr=(0, 1), indices=(0,), values=(0.0,))
    assert matrix.nnz == 1


# ------------------------------------------------------- level storage


@pytest.mark.parametrize(
    "kinds",
    [
        (DENSE, DENSE),
        (DENSE, COMPRESSED),
        (COMPRESSED, DENSE),
        (COMPRESSED, COMPRESSED),
    ],
)
@pytest.mark.parametrize("modes", [(0, 1), (1, 0)])
def test_level_storage_round_trips_every_rank2_layout(kinds, modes):
    rows = [
        [0.0, 1.5, 0.0, -2.0],
        [0.0, 0.0, 0.0, 0.0],
        [2.0, 0.0, 3.0, 0.0],
    ]
    storage = LevelTensorStorage.from_dense(rows, (3, 4), modes, kinds)
    assert storage.to_dense() == rows


def test_level_storage_round_trips_zero_shapes():
    empty_rows = LevelTensorStorage.from_dense(
        [], (0, 3), (0, 1), (COMPRESSED, COMPRESSED)
    )
    assert empty_rows.to_dense() == []
    empty_cols = LevelTensorStorage.from_dense(
        [[], []], (2, 0), (0, 1), (DENSE, COMPRESSED)
    )
    assert empty_cols.to_dense() == [[], []]


def test_level_storage_from_csr_adapter_round_trips():
    rows = [[0.0, 1.0], [2.0, 0.0], [0.0, 0.0]]
    matrix = CsrMatrix.from_dense(rows, 2)
    storage = from_csr(matrix)
    assert storage.kinds == (DENSE, COMPRESSED)
    assert storage.modes == (0, 1)
    assert storage.shape == (3, 2)
    assert storage.to_dense() == rows
    direct = LevelTensorStorage.from_dense(rows, (3, 2), (0, 1), (DENSE, COMPRESSED))
    assert storage == direct


def test_from_csr_rejects_non_csr_containers():
    with pytest.raises(LevelStorageError, match="CsrMatrix"):
        from_csr([[1.0]])


def test_level_storage_from_dense_ragged_rejected():
    with pytest.raises(LevelStorageError, match="ragged"):
        LevelTensorStorage.from_dense(
            [[1.0, 2.0], [3.0]], (2, 2), (0, 1), (DENSE, COMPRESSED)
        )


def test_level_storage_from_dense_non_numeric_rejected():
    with pytest.raises(LevelStorageError, match="numeric"):
        LevelTensorStorage.from_dense([["one"]], (1, 1), (0, 1), (DENSE, COMPRESSED))


def test_level_storage_from_dense_rejects_coordinate_levels():
    with pytest.raises(LevelStorageError, match="cannot build"):
        LevelTensorStorage.from_dense(
            [[1.0]], (1, 1), (0, 1), (LevelKind.COORDINATE, DENSE)
        )


@pytest.mark.parametrize(
    "shape,modes,kinds",
    [
        (None, (0,), (DENSE,)),
        ((1,), None, (DENSE,)),
        ((1,), (0,), None),
    ],
)
def test_level_storage_from_dense_rejects_non_iterable_metadata(shape, modes, kinds):
    with pytest.raises(LevelStorageError, match="owned list or tuple"):
        LevelTensorStorage.from_dense([1.0], shape, modes, kinds)


def test_level_storage_rejects_sequence_subclasses_without_callbacks():
    class HostileList(list):
        def __iter__(self):
            raise RuntimeError("sequence callback ran")

        def __len__(self):
            raise RuntimeError("sequence callback ran")

    with pytest.raises(LevelStorageError, match="owned list or tuple"):
        LevelTensorStorage.from_dense(
            [1.0],
            HostileList((1,)),
            (0,),
            (DENSE,),
        )
    with pytest.raises(LevelStorageError, match="ragged or mis-shaped"):
        LevelTensorStorage.from_dense(
            HostileList((1.0,)),
            (1,),
            (0,),
            (DENSE,),
        )


@pytest.mark.parametrize("modes", ((0, 2), (0, 0), (0, -1)))
def test_level_storage_from_dense_rejects_invalid_mode_permutations(modes):
    with pytest.raises(LevelStorageError, match="permutation"):
        LevelTensorStorage.from_dense(
            [[1.0]],
            (1, 1),
            modes,
            (DENSE, COMPRESSED),
        )


def test_level_storage_preserves_signed_zero_on_materialized_dense_leaves():
    all_dense = LevelTensorStorage.from_dense(
        [-0.0],
        (1,),
        (0,),
        (DENSE,),
    )
    assert math.copysign(1.0, all_dense.leaf_value(0)) == -1.0

    compressed_dense = LevelTensorStorage.from_dense(
        [[-0.0, 2.0]],
        (1, 2),
        (0, 1),
        (COMPRESSED, DENSE),
    )
    assert math.copysign(1.0, compressed_dense.leaf_value(0)) == -1.0
    assert compressed_dense.leaf_value(1) == 2.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            dict(
                shape=(2, 2),
                modes=(0, 0),
                levels=(DenseLevel(2), DenseLevel(2)),
                values=(0.0,) * 4,
            ),
            "permutation",
        ),
        (
            dict(
                shape=(2, 2),
                modes=(0, 1),
                levels=(DenseLevel(3), DenseLevel(2)),
                values=(0.0,) * 6,
            ),
            "does not match",
        ),
        (
            dict(
                shape=(2, 2),
                modes=(0, 1),
                levels=(
                    DenseLevel(2),
                    CompressedLevel((0, 1), (0, 1)),
                ),
                values=(0.0, 0.0),
            ),
            "segment offsets",
        ),
        (
            dict(
                shape=(2, 2),
                modes=(0, 1),
                levels=(
                    DenseLevel(2),
                    CompressedLevel((1, 1, 1), ()),
                ),
                values=(),
            ),
            "start at zero",
        ),
        (
            dict(
                shape=(2, 2),
                modes=(0, 1),
                levels=(
                    DenseLevel(2),
                    CompressedLevel((0, 2, 1), (0, 1)),
                ),
                values=(1.0, 1.0),
            ),
            "nondecreasing",
        ),
        (
            dict(
                shape=(2, 2),
                modes=(0, 1),
                levels=(
                    DenseLevel(2),
                    CompressedLevel((0, 1, 2), (0,)),
                ),
                values=(1.0,),
            ),
            "terminate",
        ),
        (
            dict(
                shape=(2, 2),
                modes=(0, 1),
                levels=(
                    DenseLevel(2),
                    CompressedLevel((0, 1, 2), (0, 5)),
                ),
                values=(1.0, 1.0),
            ),
            "outside",
        ),
        (
            dict(
                shape=(2, 3),
                modes=(0, 1),
                levels=(
                    DenseLevel(2),
                    CompressedLevel((0, 2, 2), (1, 0)),
                ),
                values=(1.0, 1.0),
            ),
            "strictly increasing",
        ),
        (
            dict(
                shape=(2, 2),
                modes=(0, 1),
                levels=(
                    DenseLevel(2),
                    CompressedLevel((0, 1, 1), (0,)),
                ),
                values=(1.0, 2.0),
            ),
            "leaf positions",
        ),
        (
            dict(
                shape=(2, 2),
                modes=(0, 1),
                levels=(
                    DenseLevel(2),
                    CompressedLevel((0, 1, 1), (0,)),
                ),
                values=(1,),
            ),
            "exact float",
        ),
        (
            dict(
                shape=(2, 2),
                modes=(0, 1),
                levels=(DenseLevel(2), "compressed"),
                values=(0.0,) * 4,
            ),
            "unsupported storage class",
        ),
        (
            dict(
                shape=(2, 2),
                modes=(0, 1),
                levels=(DenseLevel(2),),
                values=(0.0,) * 4,
            ),
            "length 2",
        ),
    ],
)
def test_level_storage_non_canonical_construction_rejected(kwargs, match):
    with pytest.raises(LevelStorageError, match=match):
        LevelTensorStorage(**kwargs)


def test_level_storage_accessor_boundaries():
    storage = LevelTensorStorage.from_dense(
        [[1.0, 0.0], [0.0, 2.0]], (2, 2), (0, 1), (DENSE, COMPRESSED)
    )
    with pytest.raises(LevelStorageError, match="no stored segments"):
        storage.segment(0, 0)
    with pytest.raises(LevelStorageError, match="no level 5"):
        storage.segment(5, 0)
    with pytest.raises(LevelStorageError, match="parent position"):
        storage.segment(1, 7)
    with pytest.raises(LevelStorageError, match="no stored coordinates"):
        storage.coordinate_at(0, 0)
    with pytest.raises(LevelStorageError, match="outside"):
        storage.coordinate_at(1, 9)
    with pytest.raises(LevelStorageError, match="leaf position"):
        storage.leaf_value(9)
    assert storage.segment(1, 0) == (0, 1)
    assert storage.coordinate_at(1, 1) == 1
    assert storage.leaf_value(1) == 2.0


@pytest.mark.parametrize("bad", (True, 1.0, "1"))
def test_level_storage_accessors_reject_non_exact_integer_arguments(bad):
    storage = LevelTensorStorage.from_dense(
        [[1.0]], (1, 1), (0, 1), (DENSE, COMPRESSED)
    )
    with pytest.raises(LevelStorageError, match="exact int"):
        storage.segment(bad, 0)
    with pytest.raises(LevelStorageError, match="exact int"):
        storage.segment(1, bad)
    with pytest.raises(LevelStorageError, match="exact int"):
        storage.coordinate_at(1, bad)
    with pytest.raises(LevelStorageError, match="exact int"):
        storage.leaf_value(bad)


def test_level_storage_accessors_fail_closed_on_forged_nested_state():
    storage = LevelTensorStorage.from_dense(
        [[1.0, 0.0], [0.0, 2.0]],
        (2, 2),
        (0, 1),
        (DENSE, COMPRESSED),
    )
    compressed = storage.levels[1]
    object.__setattr__(compressed, "seg_offsets", (0,))
    with pytest.raises(LevelStorageError, match="no segment"):
        storage.segment(1, 0)

    object.__setattr__(compressed, "seg_offsets", (0, 1, 2))
    object.__setattr__(compressed, "coords", (7, 1))
    with pytest.raises(LevelStorageError, match="outside"):
        storage.coordinate_at(1, 0)

    object.__delattr__(compressed, "coords")
    with pytest.raises(LevelStorageError, match="missing stored field"):
        storage.coordinate_at(1, 0)


def test_level_storage_snapshot_is_deep_and_revalidates_caller_state():
    storage = LevelTensorStorage.from_dense(
        [[2.0, 3.0], [4.0, 0.0]],
        (2, 2),
        (0, 1),
        (COMPRESSED, COMPRESSED),
    )
    snapshot = storage.snapshot()
    assert snapshot == storage
    assert snapshot is not storage
    assert all(
        copied is not original
        for copied, original in zip(snapshot.levels, storage.levels)
    )

    object.__setattr__(storage.levels[1], "coords", (0, 0, 0))
    assert snapshot.to_dense() == [[2.0, 3.0], [4.0, 0.0]]
    with pytest.raises(LevelStorageError, match="strictly increasing"):
        storage.snapshot()


@pytest.mark.parametrize("forged", ([0, 1], None))
def test_level_storage_snapshot_rejects_non_tuple_nested_streams(forged):
    storage = LevelTensorStorage.from_dense(
        [[1.0]],
        (1, 1),
        (0, 1),
        (DENSE, COMPRESSED),
    )
    object.__setattr__(storage.levels[1], "coords", forged)
    with pytest.raises(LevelStorageError, match="owned tuple"):
        storage.snapshot()


def test_level_storage_construction_rejects_missing_nested_fields():
    dense = DenseLevel(1)
    object.__delattr__(dense, "extent")
    with pytest.raises(LevelStorageError, match="missing stored field 'extent'"):
        LevelTensorStorage(
            shape=(1,),
            modes=(0,),
            levels=(dense,),
            values=(0.0,),
        )


def test_level_storage_from_dense_rejects_unknown_kinds_and_unrepresentable_values():
    with pytest.raises(LevelStorageError, match="LevelKind member"):
        LevelTensorStorage.from_dense([[1.0]], (1, 1), (0, 1), (DENSE, "dense"))
    with pytest.raises(LevelStorageError, match="representable"):
        LevelTensorStorage.from_dense(
            [[10**10000]],
            (1, 1),
            (0, 1),
            (DENSE, COMPRESSED),
        )


@pytest.mark.parametrize(
    "name,shape,match",
    [
        ("", (1, 1), "nonempty str"),
        ("C", (1,), "rank-2 tuple"),
        ("C", (1, True), "rank-2 tuple"),
    ],
)
def test_csr_output_builder_constructor_fails_closed(name, shape, match):
    with pytest.raises(LevelStorageError, match=match):
        CsrOutputBuilder(name, shape)


def test_csr_output_builder_append_fails_closed_on_bad_types():
    builder = CsrOutputBuilder("C", (1, 1))
    with pytest.raises(LevelStorageError, match="two exact ints"):
        builder.append((0,), 1.0)
    with pytest.raises(LevelStorageError, match="two exact ints"):
        builder.append((0, True), 1.0)
    with pytest.raises(LevelStorageError, match="exact float"):
        builder.append((0, 0), 1)


# --------------------------------------------------- interpreter contract


def test_missing_and_extra_bindings_fail_closed():
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[1.0]], 1)
    with pytest.raises(LoopIRInterpreterError, match="exactly the declared inputs"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix},
            {fixture.result: (1,)},
        )
    with pytest.raises(LoopIRInterpreterError, match="exactly the declared outputs"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: [1.0]},
            {fixture.result: (1,), new_symbol_id(): (1,)},
        )


def test_disappearing_mapping_values_fail_closed():
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[1.0]], 1)

    class VanishingValues(dict):
        def __getitem__(self, _key):
            raise KeyError("vanished")

    with pytest.raises(
        LoopIRInterpreterError, match="input bindings could not be snapshotted"
    ):
        run_program(
            fixture.program,
            VanishingValues({fixture.matrix: matrix, fixture.vector: [1.0]}),
            {fixture.result: (1,)},
        )
    with pytest.raises(
        LoopIRInterpreterError, match="output shapes could not be snapshotted"
    ):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: [1.0]},
            VanishingValues({fixture.result: (1,)}),
        )


def test_hostile_foreign_mapping_keys_fail_before_equality_callbacks():
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[1.0]], 1)

    class HostileKey:
        def __init__(self, collision):
            self.collision = collision

        def __hash__(self):
            return hash(self.collision)

        def __eq__(self, _other):
            raise RuntimeError("foreign key equality callback ran")

    class AdvertisedMapping:
        def __init__(self, keys):
            self.keys = keys

        def __iter__(self):
            return iter(self.keys)

        def __len__(self):
            return len(self.keys)

        def __getitem__(self, _key):
            raise AssertionError("foreign key reached value lookup")

    with pytest.raises(LoopIRInterpreterError, match="exact int-valued SymbolId"):
        run_program(
            fixture.program,
            AdvertisedMapping(
                (HostileKey(fixture.matrix), fixture.matrix, fixture.vector)
            ),
            {fixture.result: (1,)},
        )
    with pytest.raises(LoopIRInterpreterError, match="exact int-valued SymbolId"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: [1.0]},
            AdvertisedMapping((HostileKey(fixture.result), fixture.result)),
        )


def test_non_csr_sparse_binding_fails_closed():
    fixture = build_csr_spmv_program()
    with pytest.raises(LoopIRInterpreterError, match="CsrMatrix"):
        run_program(
            fixture.program,
            {fixture.matrix: [[1.0]], fixture.vector: [1.0]},
            {fixture.result: (1,)},
        )


def test_level_storage_bound_to_csr_declaration_fails_closed():
    fixture = build_csr_spmv_program()
    storage = matrix_storage([[1.0]], 1, (0, 1), (DENSE, COMPRESSED))
    with pytest.raises(LoopIRInterpreterError, match="CsrMatrix"):
        run_program(
            fixture.program,
            {fixture.matrix: storage, fixture.vector: [1.0]},
            {fixture.result: (1,)},
        )


def test_csr_container_bound_to_dcsr_declaration_fails_closed():
    fixture = build_dcsr_spmv_program()
    matrix = CsrMatrix.from_dense([[1.0]], 1)
    with pytest.raises(LoopIRInterpreterError, match="LevelTensorStorage"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: [1.0]},
            {fixture.result: (1,)},
        )


def test_mismatched_storage_layout_fails_closed():
    fixture = build_dcsr_spmv_program()
    csc_storage = matrix_storage([[1.0]], 1, (1, 0), (DENSE, COMPRESSED))
    with pytest.raises(LoopIRInterpreterError, match="does not match"):
        run_program(
            fixture.program,
            {fixture.matrix: csc_storage, fixture.vector: [1.0]},
            {fixture.result: (1,)},
        )


def test_interpreter_snapshots_level_storage_before_execution():
    fixture = build_dcsr_spmv_program()
    storage = matrix_storage(
        [[2.0, 3.0], [4.0, 0.0]],
        2,
        (0, 1),
        (COMPRESSED, COMPRESSED),
    )
    interpreter = loopir_interp._Interpreter(
        fixture.program,
        {fixture.matrix: storage, fixture.vector: [10.0, 100.0]},
        {fixture.result: (2,)},
    )

    object.__setattr__(storage.levels[1], "coords", (0, 0, 0))
    assert interpreter.run()[fixture.result] == [320.0, 40.0]


def test_interpreter_rejects_forged_level_storage_before_execution():
    fixture = build_dcsr_spmv_program()
    storage = matrix_storage(
        [[2.0, 3.0], [4.0, 0.0]],
        2,
        (0, 1),
        (COMPRESSED, COMPRESSED),
    )
    object.__setattr__(storage.levels[1], "coords", (0, 0, 0))
    with pytest.raises(LoopIRInterpreterError, match="invalid level storage"):
        run_program(
            fixture.program,
            {fixture.matrix: storage, fixture.vector: [10.0, 100.0]},
            {fixture.result: (2,)},
        )


@pytest.mark.parametrize("forgery", ("duplicate_indices", "missing_n_rows"))
def test_interpreter_rejects_forged_csr_before_execution(forgery):
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[2.0, 3.0]], 2)
    if forgery == "duplicate_indices":
        object.__setattr__(matrix, "indices", (0, 0))
    else:
        object.__delattr__(matrix, "n_rows")
    with pytest.raises(LoopIRInterpreterError, match="invalid CSR storage"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: [10.0, 100.0]},
            {fixture.result: (1,)},
        )


def test_explicit_all_dense_storage_preserves_empty_outer_shape():
    d0 = new_dimension_id()
    d1 = new_dimension_id()
    x = new_symbol_id()
    y = new_symbol_id()
    j = new_index_id()
    program = LoopProgram(
        nid(),
        (DimensionDecl(nid(), d0, "empty"), DimensionDecl(nid(), d1, "inner")),
        (
            TensorDecl(
                nid(),
                x,
                "x",
                (d0, d1),
                (LevelDecl(nid(), DENSE, 0), LevelDecl(nid(), DENSE, 1)),
            ),
            TensorDecl(nid(), y, "y", (d1,), (LevelDecl(nid(), DENSE, 0),)),
        ),
        (x,),
        (y,),
        Block(
            nid(),
            (
                DenseFor(
                    nid(),
                    j,
                    d1,
                    Block(
                        nid(),
                        (
                            Store(
                                nid(),
                                y,
                                (IndexValue(nid(), j),),
                                FloatConst(nid(), 1.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    storage = LevelTensorStorage.from_dense(
        [],
        (0, 3),
        (0, 1),
        (DENSE, DENSE),
    )
    assert run_program(program, {x: storage}, {y: (3,)})[y] == [1.0, 1.0, 1.0]


def test_position_load_lazily_materializes_a_nested_dense_binding():
    d = new_dimension_id()
    x = new_symbol_id()
    y = new_symbol_id()
    i = new_index_id()
    index = IndexValue(nid(), i)
    program = LoopProgram(
        nid(),
        (DimensionDecl(nid(), d, "d"),),
        (
            TensorDecl(nid(), x, "x", (d,), (LevelDecl(nid(), DENSE, 0),)),
            TensorDecl(nid(), y, "y", (d,), (LevelDecl(nid(), DENSE, 0),)),
        ),
        (x,),
        (y,),
        Block(
            nid(),
            (
                DenseFor(
                    nid(),
                    i,
                    d,
                    Block(
                        nid(),
                        (
                            Store(
                                nid(),
                                y,
                                (IndexValue(nid(), i),),
                                PositionLoad(
                                    nid(),
                                    x,
                                    DensePosition(
                                        nid(),
                                        x,
                                        0,
                                        RootPosition(nid()),
                                        index,
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    result = run_program(program, {x: [-0.0, 3.0]}, {y: (2,)})[y]
    assert math.copysign(1.0, result[0]) == -1.0
    assert result[1] == 3.0


def test_ordinary_dense_load_does_not_build_level_storage(monkeypatch):
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[2.0]], 1)

    def unexpected_level_materialization(*_args, **_kwargs):
        raise AssertionError("ordinary dense Load built physical level storage")

    monkeypatch.setattr(
        LevelTensorStorage,
        "from_dense",
        unexpected_level_materialization,
    )
    result = run_program(
        fixture.program,
        {fixture.matrix: matrix, fixture.vector: [3.0]},
        {fixture.result: (1,)},
    )
    assert result[fixture.result] == [6.0]


def test_non_csr_sparse_output_layout_fails_closed():
    di, dj = new_dimension_id(), new_dimension_id()
    result = new_symbol_id()
    program = LoopProgram(
        nid(),
        (DimensionDecl(nid(), di, "i"), DimensionDecl(nid(), dj, "j")),
        (
            TensorDecl(
                nid(),
                result,
                "C",
                (di, dj),
                (LevelDecl(nid(), DENSE, 1), LevelDecl(nid(), COMPRESSED, 0)),
            ),
        ),
        (),
        (result,),
        Block(
            nid(),
            (
                AppendEntry(
                    nid(),
                    result,
                    (IntConst(nid(), 0), IntConst(nid(), 0)),
                    FloatConst(nid(), 1.0),
                ),
            ),
        ),
    )
    with pytest.raises(
        LoopIRInterpreterError, match="unsupported sparse output layout"
    ):
        run_program(program, {}, {result: (1, 1)})


def test_malformed_output_shape_fails_closed():
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[1.0]], 1)
    for bad_shape in ((1, 1), (-1,), (1.0,), [1]):
        with pytest.raises(LoopIRInterpreterError):
            run_program(
                fixture.program,
                {fixture.matrix: matrix, fixture.vector: [1.0]},
                {fixture.result: bad_shape},
            )


def test_non_numeric_dense_binding_fails_closed():
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[1.0]], 1)
    with pytest.raises(LoopIRInterpreterError, match="non-numeric"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: ["one"]},
            {fixture.result: (1,)},
        )


def test_unrepresentable_dense_binding_fails_closed():
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[1.0]], 1)
    with pytest.raises(LoopIRInterpreterError, match="unrepresentable numeric"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: [10**10000]},
            {fixture.result: (1,)},
        )


def test_dense_sequence_subclass_fails_without_callbacks():
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[1.0]], 1)

    class HostileList(list):
        def __len__(self):
            raise RuntimeError("sequence callback ran")

    with pytest.raises(LoopIRInterpreterError, match="must nest sequences"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: HostileList((1.0,))},
            {fixture.result: (1,)},
        )


def test_shorter_second_operand_fails_closed():
    fixture = build_csr_union_add_program()
    a = CsrMatrix.from_dense([[1.0], [2.0], [3.0]], 1)
    b = CsrMatrix.from_dense([[1.0], [2.0]], 1)
    with pytest.raises(LoopIRInterpreterError, match="dimension extent mismatch"):
        run_program(
            fixture.program,
            {fixture.lhs: a, fixture.rhs: b},
            {fixture.result: (3, 1)},
        )


@pytest.mark.parametrize("vector", ([4.0], [4.0, 0.0, 0.0, 0.0]))
def test_spmv_vector_extent_mismatch_is_sparsity_independent(vector):
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[2.0, 0.0, 0.0]], 3)
    with pytest.raises(LoopIRInterpreterError, match="dimension extent mismatch"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: vector},
            {fixture.result: (1,)},
        )


def test_spmv_output_extent_mismatch_rejected_before_execution():
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[0.0, 0.0, 0.0]], 3)
    with pytest.raises(LoopIRInterpreterError, match="dimension extent mismatch"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: [0.0, 0.0, 0.0]},
            {fixture.result: (2,)},
        )


def test_dcsr_storage_extent_mismatch_is_sparsity_independent():
    fixture = build_dcsr_spmv_program()
    storage = matrix_storage([[0.0, 0.0, 0.0]], 3, (0, 1), (COMPRESSED, COMPRESSED))
    with pytest.raises(LoopIRInterpreterError, match="dimension extent mismatch"):
        run_program(
            fixture.program,
            {fixture.matrix: storage, fixture.vector: [1.0, 2.0]},
            {fixture.result: (1,)},
        )


def test_extent_mismatch_rejected_before_any_tensor_materialization(monkeypatch):
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[0.0]], 1)

    def unexpected_materialization(*_args, **_kwargs):
        raise AssertionError("tensor materialized before extent validation")

    monkeypatch.setattr(
        loopir_interp._Interpreter,
        "_materialize_input",
        unexpected_materialization,
    )
    monkeypatch.setattr(
        loopir_interp._Interpreter,
        "_materialize_output",
        unexpected_materialization,
    )
    with pytest.raises(LoopIRInterpreterError, match="dimension extent mismatch"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: [0.0]},
            {fixture.result: (2,)},
        )


def test_input_mapping_is_snapshotted_once_before_shape_validation():
    fixture = build_csr_spmv_program()
    checked_matrix = CsrMatrix.from_dense([[2.0]], 1)
    swapped_matrix = CsrMatrix.from_dense([[9.0, 0.0], [0.0, 0.0]], 2)

    class ChangingBindings(dict):
        def __init__(self):
            super().__init__({fixture.matrix: checked_matrix, fixture.vector: [3.0]})
            self.lookups = {fixture.matrix: 0, fixture.vector: 0}

        def __getitem__(self, symbol):
            self.lookups[symbol] += 1
            if symbol == fixture.matrix and self.lookups[symbol] > 1:
                return swapped_matrix
            return super().__getitem__(symbol)

    bindings = ChangingBindings()
    results = run_program(
        fixture.program,
        bindings,
        {fixture.result: (1,)},
    )
    assert results[fixture.result] == [6.0]
    assert bindings.lookups == {fixture.matrix: 1, fixture.vector: 1}


def test_elementwise_extra_rhs_rows_rejected_before_execution():
    fixture = build_csr_union_add_program()
    lhs = CsrMatrix.from_dense([[0.0]], 1)
    rhs = CsrMatrix.from_dense([[0.0], [9.0]], 1)
    with pytest.raises(LoopIRInterpreterError, match="dimension extent mismatch"):
        run_program(
            fixture.program,
            {fixture.lhs: lhs, fixture.rhs: rhs},
            {fixture.result: (1, 1)},
        )


def test_elementwise_column_mismatch_rejected_with_empty_support():
    fixture = build_csr_intersection_multiply_program()
    lhs = CsrMatrix.from_dense([[0.0, 0.0, 0.0]], 3)
    rhs = CsrMatrix.from_dense([[0.0, 0.0]], 2)
    with pytest.raises(LoopIRInterpreterError, match="dimension extent mismatch"):
        run_program(
            fixture.program,
            {fixture.lhs: lhs, fixture.rhs: rhs},
            {fixture.result: (1, 3)},
        )


def test_elementwise_output_extent_mismatch_rejected_with_empty_support():
    fixture = build_csr_union_add_program()
    lhs = CsrMatrix.from_dense([], 0)
    rhs = CsrMatrix.from_dense([], 0)
    with pytest.raises(LoopIRInterpreterError, match="dimension extent mismatch"):
        run_program(
            fixture.program,
            {fixture.lhs: lhs, fixture.rhs: rhs},
            {fixture.result: (0, 1)},
        )


def _append_only_program(coords_list):
    di, dj = new_dimension_id(), new_dimension_id()
    result = new_symbol_id()
    statements = tuple(
        AppendEntry(
            nid(),
            result,
            (IntConst(nid(), row), IntConst(nid(), column)),
            FloatConst(nid(), 1.0),
        )
        for row, column in coords_list
    )
    program = LoopProgram(
        nid(),
        (DimensionDecl(nid(), di, "i"), DimensionDecl(nid(), dj, "j")),
        (
            TensorDecl(
                nid(),
                result,
                "C",
                (di, dj),
                (LevelDecl(nid(), DENSE, 0), LevelDecl(nid(), COMPRESSED, 1)),
            ),
        ),
        (),
        (result,),
        Block(nid(), statements),
    )
    return program, result


def test_out_of_order_append_fails_closed():
    program, result = _append_only_program([(0, 1), (0, 0)])
    with pytest.raises(LoopIRInterpreterError, match="lexicographically"):
        run_program(program, {}, {result: (1, 2)})


def test_duplicate_coordinate_append_fails_closed():
    program, result = _append_only_program([(0, 1), (0, 1)])
    with pytest.raises(LoopIRInterpreterError, match="lexicographically"):
        run_program(program, {}, {result: (1, 2)})


def test_append_outside_declared_shape_fails_closed():
    program, result = _append_only_program([(0, 5)])
    with pytest.raises(LoopIRInterpreterError, match="escapes shape"):
        run_program(program, {}, {result: (1, 2)})


def test_ordered_appends_assemble_canonical_output():
    program, result = _append_only_program([(0, 1), (1, 0), (2, 2)])
    results = run_program(program, {}, {result: (4, 3)})
    out = results[result]
    assert type(out) is CsrMatrix
    assert out.indptr == (0, 1, 2, 3, 3)
    assert out.indices == (1, 0, 2)


def _store_at_program(index):
    dx, dy = new_dimension_id(), new_dimension_id()
    x = new_symbol_id()
    y = new_symbol_id()
    program = LoopProgram(
        nid(),
        (DimensionDecl(nid(), dx, "n"), DimensionDecl(nid(), dy, "m")),
        (
            TensorDecl(nid(), x, "x", (dx,), (LevelDecl(nid(), DENSE, 0),)),
            TensorDecl(nid(), y, "y", (dy,), (LevelDecl(nid(), DENSE, 0),)),
        ),
        (x,),
        (y,),
        Block(
            nid(),
            (
                Store(
                    nid(),
                    y,
                    (IntConst(nid(), index),),
                    FloatConst(nid(), 1.0),
                ),
            ),
        ),
    )
    return program, x, y


def test_store_out_of_bounds_fails_closed():
    program, x, y = _store_at_program(5)
    with pytest.raises(LoopIRInterpreterError, match="out of bounds"):
        run_program(program, {x: [1.0]}, {y: (2,)})


def test_unresolved_dimension_extent_fails_closed():
    from scorch.compiler.loopir_spike.verifier import LoopIRVerificationError

    dy = new_dimension_id()
    unresolved = new_dimension_id()
    y = new_symbol_id()
    i = new_index_id()
    program = LoopProgram(
        nid(),
        (
            DimensionDecl(nid(), dy, "m"),
            DimensionDecl(nid(), unresolved, "ghost"),
        ),
        (TensorDecl(nid(), y, "y", (dy,), (LevelDecl(nid(), DENSE, 0),)),),
        (),
        (y,),
        Block(
            nid(),
            (
                DenseFor(
                    nid(),
                    i,
                    unresolved,
                    Block(
                        nid(),
                        (
                            Store(
                                nid(),
                                y,
                                (IntConst(nid(), 0),),
                                FloatConst(nid(), 1.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(LoopIRVerificationError) as excinfo:
        run_program(program, {}, {y: (1,)})
    assert excinfo.value.defect.code == "unresolved_dimension"


def test_wrong_parent_program_is_rejected_before_execution():
    from scorch.compiler.loopir_spike.verifier import LoopIRVerificationError

    di, dj = new_dimension_id(), new_dimension_id()
    a, y = new_symbol_id(), new_symbol_id()
    j = new_index_id()
    cursor = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=a,
        level=1,
        parent=RootPosition(nid()),
    )
    program = LoopProgram(
        nid(),
        (DimensionDecl(nid(), di, "i"), DimensionDecl(nid(), dj, "j")),
        (
            TensorDecl(
                nid(),
                a,
                "A",
                (di, dj),
                (LevelDecl(nid(), DENSE, 0), LevelDecl(nid(), COMPRESSED, 1)),
            ),
            TensorDecl(nid(), y, "y", (di,), (LevelDecl(nid(), DENSE, 0),)),
        ),
        (a,),
        (y,),
        Block(
            nid(),
            (
                SparseFor(nid(), cursor, new_position_id(), j, Block(nid(), ())),
                Store(nid(), y, (IntConst(nid(), 0),), FloatConst(nid(), 1.0)),
            ),
        ),
    )
    with pytest.raises(LoopIRVerificationError) as excinfo:
        run_program(
            program,
            {a: CsrMatrix.from_dense([[1.0]], 1)},
            {y: (1,)},
        )
    assert excinfo.value.defect.code == "parent_position_mismatch"


def test_run_program_verifies_first():
    from scorch.compiler.loopir_spike.verifier import LoopIRVerificationError

    program, result = _append_only_program([(0, 0)])
    forged = LoopProgram(
        program.node_id,
        program.dimensions,
        program.tensors,
        program.inputs,
        (),
        program.body,
    )
    with pytest.raises(LoopIRVerificationError):
        run_program(forged, {}, {})


def test_output_shape_resolves_loop_dimension():
    d = new_dimension_id()
    y = new_symbol_id()
    i = new_index_id()
    program = LoopProgram(
        nid(),
        (DimensionDecl(nid(), d, "d"),),
        (TensorDecl(nid(), y, "y", (d,), (LevelDecl(nid(), DENSE, 0),)),),
        (),
        (y,),
        Block(
            nid(),
            (
                DenseFor(
                    nid(),
                    i,
                    d,
                    Block(
                        nid(),
                        (
                            Store(
                                nid(),
                                y,
                                (IndexValue(nid(), i),),
                                FloatConst(nid(), 2.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    results = run_program(program, {}, {y: (3,)})
    assert results[y] == [2.0, 2.0, 2.0]
