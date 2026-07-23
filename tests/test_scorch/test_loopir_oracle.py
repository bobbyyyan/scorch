"""Differential and fail-closed coverage for the production LoopIR oracle.

The oracle executes verified dense programs over plain Python containers and
is compared exactly against independent pure-Python references — both sides
use Python-float arithmetic in the same order, so no tolerances are involved.
"""

import random

import pytest

from scorch.compiler.identity import SymbolId
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.nodes import BinaryOp, ReduceOp, ScalarType
from scorch.compiler.loopir.oracle import (
    MAX_ORACLE_RANK,
    LoopIROracleError,
    run_program,
)
from scorch.compiler.loopir.verifier import verify_program

from tests.test_scorch.test_loopir_printer import build_matvec
from tests.test_scorch.test_loopir_verifier import (
    build_matmul,
    build_vector_add,
    forge,
)


def reference_vector_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def reference_matvec(matrix, vector):
    return [
        sum(matrix[i][j] * vector[j] for j in range(len(vector)))
        for i in range(len(matrix))
    ]


def reference_matmul(a, b):
    rows, inner, cols = len(a), len(b), len(b[0]) if b else 0
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            for j in range(cols):
                result[i][j] += a[i][k] * b[k][j]
    return result


def test_vector_add_matches_reference_exactly():
    fixture = build_vector_add()
    a = [1.5, -2.0, 0.25, 8.0]
    b = [0.5, 4.0, -0.75, 1.0]
    results = run_program(
        fixture.program,
        {fixture.a: a, fixture.b: b},
        {fixture.c: (4,)},
    )
    assert results[fixture.c] == reference_vector_add(a, b)


def test_matvec_matches_reference_exactly():
    program = build_matvec()
    a_symbol, x_symbol = program.inputs
    y_symbol = program.outputs[0]
    matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    vector = [0.5, -1.0, 2.0]
    results = run_program(
        program,
        {a_symbol: matrix, x_symbol: vector},
        {y_symbol: (2,)},
    )
    assert results[y_symbol] == reference_matvec(matrix, vector)


def test_matmul_matches_reference_exactly_on_random_grids():
    rng = random.Random(20260722)
    for rows, inner, cols in [(1, 1, 1), (2, 3, 4), (5, 2, 3), (3, 4, 1)]:
        fixture = build_matmul()
        a = [[rng.uniform(-2, 2) for _ in range(inner)] for _ in range(rows)]
        b = [[rng.uniform(-2, 2) for _ in range(cols)] for _ in range(inner)]
        results = run_program(
            fixture.program,
            {fixture.a: a, fixture.b: b},
            {fixture.c: (rows, cols)},
        )
        assert results[fixture.c] == reference_matmul(a, b)


def test_zero_extent_shapes_execute():
    # Hidden suffix extents are resolved through shared DimensionIds before
    # the nested values are copied.
    program = build_matvec()
    a_symbol, x_symbol = program.inputs
    y_symbol = program.outputs[0]
    results = run_program(
        program,
        {a_symbol: [[], []], x_symbol: []},
        {y_symbol: (2,)},
    )
    assert results[y_symbol] == [0.0, 0.0]

    fixture = build_vector_add()
    results = run_program(
        fixture.program,
        {fixture.a: [], fixture.b: []},
        {fixture.c: (0,)},
    )
    assert results[fixture.c] == []


def test_zero_leading_extents_resolve_across_tensor_dimensions():
    fixture = build_matmul()
    results = run_program(
        fixture.program,
        {fixture.a: [], fixture.b: []},
        {fixture.c: (0, 0)},
    )
    assert results[fixture.c] == []


def test_rank_three_intermediate_zero_extent_resolves_from_output():
    builder = LoopIRBuilder()
    dimensions = tuple(builder.dimension(name) for name in ("i", "j", "k"))
    a, b, c = (builder.new_symbol_id() for _ in range(3))
    dimension_ids = tuple(decl.dimension for decl in dimensions)
    tensors = tuple(
        builder.tensor(
            symbol,
            name,
            ScalarType.FLOAT32,
            dimension_ids,
            builder.dense_levels(3),
        )
        for symbol, name in ((a, "A"), (b, "B"), (c, "C"))
    )
    indices = tuple(builder.new_index_id() for _ in range(3))

    def access_indices():
        return tuple(builder.index_value(index) for index in indices)

    store = builder.store(
        c,
        access_indices(),
        builder.binary(
            BinaryOp.ADD,
            builder.load(a, access_indices()),
            builder.load(b, access_indices()),
        ),
    )
    body = builder.block((store,))
    for index, dimension in reversed(tuple(zip(indices, dimension_ids))):
        body = builder.block((builder.dense_for(index, dimension, body),))
    program = builder.program(dimensions, tensors, (a, b), (c,), body)

    results = run_program(
        program,
        {a: [[], []], b: [[], []]},
        {c: (2, 0, 3)},
    )
    assert results[c] == [[], []]


def test_unresolved_hidden_zero_extent_fails_closed():
    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, y = builder.new_symbol_id(), builder.new_symbol_id()
    decl_a = builder.tensor(
        a,
        "A",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        builder.dense_levels(2),
    )
    decl_y = builder.tensor(
        y,
        "y",
        ScalarType.FLOAT32,
        (dim_i.dimension,),
        builder.dense_levels(1),
    )
    index_i, index_j = builder.new_index_id(), builder.new_index_id()
    leaf = builder.store_reduce(
        y,
        (builder.index_value(index_i),),
        ReduceOp.ADD,
        builder.load(
            a,
            (builder.index_value(index_i), builder.index_value(index_j)),
        ),
    )
    inner = builder.dense_for(index_j, dim_j.dimension, builder.block((leaf,)))
    outer = builder.dense_for(index_i, dim_i.dimension, builder.block((inner,)))
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_y),
        (a,),
        (y,),
        builder.block((outer,)),
    )
    with pytest.raises(LoopIROracleError, match="cannot be inferred at mode 1"):
        run_program(program, {a: []}, {y: (0,)})


def test_excessive_dense_rank_fails_closed_before_recursive_storage_work():
    rank = MAX_ORACLE_RANK + 1
    builder = LoopIRBuilder()
    dimension = builder.dimension("d")
    a, c = builder.new_symbol_id(), builder.new_symbol_id()
    dimensions = (dimension.dimension,) * rank
    a_decl = builder.tensor(
        a, "A", ScalarType.FLOAT32, dimensions, builder.dense_levels(rank)
    )
    c_decl = builder.tensor(
        c, "C", ScalarType.FLOAT32, dimensions, builder.dense_levels(rank)
    )
    index = builder.new_index_id()

    def access_indices():
        return tuple(builder.index_value(index) for _ in range(rank))

    store = builder.store(c, access_indices(), builder.load(a, access_indices()))
    program = builder.program(
        (dimension,),
        (a_decl, c_decl),
        (a,),
        (c,),
        builder.block(
            (
                builder.dense_for(
                    index,
                    dimension.dimension,
                    builder.block((store,)),
                ),
            )
        ),
    )
    verify_program(program)
    with pytest.raises(LoopIROracleError, match="rank .* exceeds"):
        run_program(program, {a: []}, {c: (0,) * rank})


def test_outputs_are_zero_initialized_before_reduction():
    program = build_matvec()
    a_symbol, x_symbol = program.inputs
    y_symbol = program.outputs[0]
    results = run_program(
        program,
        {a_symbol: [[0.0, 0.0]], x_symbol: [3.0, 4.0]},
        {y_symbol: (1,)},
    )
    assert results[y_symbol] == [0.0]


def test_sub_semantics():
    fixture = build_vector_add()
    store = fixture.program.body.statements[0].body.statements[0]
    forge(store.value, op=BinaryOp.SUB)
    results = run_program(
        fixture.program,
        {fixture.a: [5.0, 1.0], fixture.b: [2.0, 7.0]},
        {fixture.c: (2,)},
    )
    assert results[fixture.c] == [3.0, -6.0]


def test_input_bindings_must_cover_declared_inputs():
    fixture = build_vector_add()
    with pytest.raises(LoopIROracleError):
        run_program(fixture.program, {fixture.a: [1.0]}, {fixture.c: (1,)})
    with pytest.raises(LoopIROracleError):
        run_program(
            fixture.program,
            {fixture.a: [1.0], fixture.b: [1.0], SymbolId(31_337): [1.0]},
            {fixture.c: (1,)},
        )


def test_output_shapes_must_cover_declared_outputs():
    fixture = build_vector_add()
    with pytest.raises(LoopIROracleError):
        run_program(fixture.program, {fixture.a: [1.0], fixture.b: [1.0]}, {})


def test_ragged_input_fails_closed():
    fixture = build_matmul()
    with pytest.raises(LoopIROracleError):
        run_program(
            fixture.program,
            {fixture.a: [[1.0, 2.0], [3.0]], fixture.b: [[1.0], [1.0]]},
            {fixture.c: (2, 1)},
        )


def test_non_numeric_input_fails_closed():
    fixture = build_vector_add()
    with pytest.raises(LoopIROracleError):
        run_program(
            fixture.program,
            {fixture.a: ["1.0"], fixture.b: [1.0]},
            {fixture.c: (1,)},
        )


def test_extent_mismatch_fails_before_execution():
    fixture = build_vector_add()
    with pytest.raises(LoopIROracleError) as error:
        run_program(
            fixture.program,
            {fixture.a: [1.0, 2.0], fixture.b: [1.0]},
            {fixture.c: (2,)},
        )
    assert "dimension extent mismatch" in str(error.value)


def test_output_extent_participates_in_dimension_resolution():
    fixture = build_vector_add()
    with pytest.raises(LoopIROracleError) as error:
        run_program(
            fixture.program,
            {fixture.a: [1.0, 2.0], fixture.b: [1.0, 2.0]},
            {fixture.c: (3,)},
        )
    assert "dimension extent mismatch" in str(error.value)


def test_invalid_output_shape_fails_closed():
    fixture = build_vector_add()
    with pytest.raises(LoopIROracleError):
        run_program(
            fixture.program,
            {fixture.a: [1.0], fixture.b: [1.0]},
            {fixture.c: (1, 1)},
        )
    with pytest.raises(LoopIROracleError):
        run_program(
            fixture.program,
            {fixture.a: [1.0], fixture.b: [1.0]},
            {fixture.c: (-1,)},
        )


def test_oracle_verifies_first():
    fixture = build_vector_add()
    forge(fixture.program.tensors[1], dtype=ScalarType.FLOAT64)
    from scorch.compiler.loopir.verifier import LoopIRVerificationError

    with pytest.raises(LoopIRVerificationError):
        run_program(
            fixture.program,
            {fixture.a: [1.0], fixture.b: [1.0]},
            {fixture.c: (1,)},
        )


def test_oracle_does_not_mutate_inputs():
    fixture = build_vector_add()
    a = [1.0, 2.0]
    b = [3.0, 4.0]
    run_program(fixture.program, {fixture.a: a, fixture.b: b}, {fixture.c: (2,)})
    assert a == [1.0, 2.0] and b == [3.0, 4.0]


def test_oracle_owns_inputs_before_consulting_output_shape_mapping():
    fixture = build_vector_add()
    a = [1.0]
    b = [2.0]

    class MutatingOutputShapes(dict):
        def __getitem__(self, key):
            a[0] = 10.0
            return super().__getitem__(key)

    results = run_program(
        fixture.program,
        {fixture.a: a, fixture.b: b},
        MutatingOutputShapes({fixture.c: (1,)}),
    )
    assert results[fixture.c] == [3.0]
    assert a == [10.0]


# -- Phase-5 sparse subset ----------------------------------------------------

from scorch.compiler.loopir.build import LoopIRBuilder as _Builder  # noqa: E402
from scorch.compiler.loopir.levels import (  # noqa: E402
    CsrMatrix,
    LevelTensorStorage,
)
from scorch.compiler.loopir.nodes import LevelKind  # noqa: E402

from tests.test_scorch.test_loopir_verifier import (  # noqa: E402
    build_csr_spmv,
    build_union_add,
)
from scorch.compiler.loopir.nodes import MergeMode  # noqa: E402


def csr_from_dense(dense):
    return CsrMatrix.from_dense(dense)


def reference_spmv_stored_order(dense, vector):
    result = [0.0] * len(dense)
    for i, row in enumerate(dense):
        for j, entry in enumerate(row):
            if entry != 0.0:
                result[i] = result[i] + entry * vector[j]
    return result


def reference_union_add(a, b):
    rows = []
    for i in range(len(a)):
        row_entries = []
        for j in range(len(a[0])):
            left, right = a[i][j], b[i][j]
            if left != 0.0 and right != 0.0:
                row_entries.append((j, left + right))
            elif left != 0.0:
                row_entries.append((j, left))
            elif right != 0.0:
                row_entries.append((j, right))
        rows.append(row_entries)
    return rows


def reference_intersection_mul(a, b):
    rows = []
    for i in range(len(a)):
        row_entries = []
        for j in range(len(a[0])):
            if a[i][j] != 0.0 and b[i][j] != 0.0:
                row_entries.append((j, a[i][j] * b[i][j]))
        rows.append(row_entries)
    return rows


def csr_rows(matrix):
    rows = []
    for i in range(matrix.n_rows):
        rows.append(
            [
                (matrix.indices[p], matrix.values[p])
                for p in range(matrix.indptr[i], matrix.indptr[i + 1])
            ]
        )
    return rows


def test_csr_spmv_matches_reference_exactly():
    fixture = build_csr_spmv()
    dense = [
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, -3.5, 0.0, 4.0],
    ]
    vector = [0.5, 1.5, -2.0, 3.0]
    results = run_program(
        fixture.program,
        {fixture.a: csr_from_dense(dense), fixture.x: vector},
        {fixture.y: (3,)},
    )
    assert results[fixture.y] == reference_spmv_stored_order(dense, vector)


def test_csr_spmv_accepts_equivalent_level_storage():
    fixture = build_csr_spmv()
    dense = [[0.0, 2.0], [3.0, 0.0]]
    vector = [4.0, 5.0]
    via_adapter = run_program(
        fixture.program,
        {fixture.a: csr_from_dense(dense), fixture.x: vector},
        {fixture.y: (2,)},
    )
    fixture2 = build_csr_spmv()
    storage = LevelTensorStorage.from_dense(
        dense, (2, 2), (0, 1), (LevelKind.DENSE, LevelKind.COMPRESSED)
    )
    via_storage = run_program(
        fixture2.program,
        {fixture2.a: storage, fixture2.x: vector},
        {fixture2.y: (2,)},
    )
    assert via_adapter[fixture.y] == via_storage[fixture2.y]


def test_union_add_covers_disjoint_overlap_and_exhaustion():
    fixture = build_union_add()
    a = [
        [1.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 7.0],
    ]
    b = [
        [0.0, 3.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 6.0],
        [2.0, 0.0, 0.0, 0.0],
    ]
    results = run_program(
        fixture.program,
        {fixture.a: csr_from_dense(a), fixture.b: csr_from_dense(b)},
        {fixture.c: (4, 4)},
    )
    produced = results[fixture.c]
    assert type(produced) is CsrMatrix
    assert csr_rows(produced) == reference_union_add(a, b)


def test_union_add_stores_explicit_zero_on_cancellation():
    fixture = build_union_add()
    a = [[2.5, 0.0]]
    b = [[-2.5, 1.0]]
    results = run_program(
        fixture.program,
        {fixture.a: csr_from_dense(a), fixture.b: csr_from_dense(b)},
        {fixture.c: (1, 2)},
    )
    produced = results[fixture.c]
    assert produced.indices == (0, 1)
    assert produced.values == (0.0, 1.0)


def test_intersection_multiply_is_structural():
    fixture = build_union_add(mode=MergeMode.INTERSECTION, with_defaults=False)
    a = [
        [1.0, 2.0, 0.0],
        [0.0, 3.0, 0.0],
    ]
    b = [
        [4.0, 0.0, 5.0],
        [0.0, 6.0, 7.0],
    ]
    results = run_program(
        fixture.program,
        {fixture.a: csr_from_dense(a), fixture.b: csr_from_dense(b)},
        {fixture.c: (2, 3)},
    )
    assert csr_rows(results[fixture.c]) == reference_intersection_mul(a, b)


def test_sparse_families_match_references_on_random_grids():
    rng = random.Random(20260722)
    for rows, cols in [(1, 1), (3, 5), (6, 4), (5, 8)]:
        a = [
            [
                rng.uniform(-2.0, 2.0) if rng.random() < 0.45 else 0.0
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]
        b = [
            [
                rng.uniform(-2.0, 2.0) if rng.random() < 0.45 else 0.0
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]
        vector = [rng.uniform(-2.0, 2.0) for _ in range(cols)]

        spmv = build_csr_spmv()
        spmv_result = run_program(
            spmv.program,
            {spmv.a: csr_from_dense(a), spmv.x: vector},
            {spmv.y: (rows,)},
        )
        assert spmv_result[spmv.y] == reference_spmv_stored_order(a, vector)

        union = build_union_add()
        union_result = run_program(
            union.program,
            {union.a: csr_from_dense(a), union.b: csr_from_dense(b)},
            {union.c: (rows, cols)},
        )
        assert csr_rows(union_result[union.c]) == reference_union_add(a, b)

        both = build_union_add(mode=MergeMode.INTERSECTION, with_defaults=False)
        both_result = run_program(
            both.program,
            {both.a: csr_from_dense(a), both.b: csr_from_dense(b)},
            {both.c: (rows, cols)},
        )
        assert csr_rows(both_result[both.c]) == reference_intersection_mul(a, b)


def test_empty_inputs_and_zero_extents_execute():
    fixture = build_union_add()
    empty = CsrMatrix(n_rows=0, n_cols=3, indptr=(0,), indices=(), values=())
    results = run_program(
        fixture.program,
        {fixture.a: empty, fixture.b: empty},
        {fixture.c: (0, 3)},
    )
    assert results[fixture.c].indptr == (0,)

    spmv = build_csr_spmv()
    no_columns = CsrMatrix(n_rows=2, n_cols=0, indptr=(0, 0, 0), indices=(), values=())
    spmv_result = run_program(
        spmv.program,
        {spmv.a: no_columns, spmv.x: []},
        {spmv.y: (2,)},
    )
    assert spmv_result[spmv.y] == [0.0, 0.0]


def test_dcsr_positions_diverge_from_coordinates():
    """Bound parent positions, not row coordinates, select DCSR segments."""

    builder = _Builder()
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
    inner = builder.sparse_for(
        builder.sparse_cursor(cursor_cols, a, 1, builder.position_value(position_rows)),
        position_cols,
        col,
        builder.block(
            (
                builder.store_reduce(
                    y,
                    (builder.index_value(row),),
                    ReduceOp.ADD,
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
    dense = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 2.5],
        [0.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
    ]
    storage = LevelTensorStorage.from_dense(
        dense, (5, 3), (0, 1), (LevelKind.COMPRESSED, LevelKind.COMPRESSED)
    )
    # Row 2 is the first stored row: its position (0) differs from its
    # coordinate (2), which is exactly what the bound parent position keys.
    assert storage.coordinate_at(0, 0) == 2
    results = run_program(program, {a: storage}, {y: (5,)})
    assert results[y] == [0.0, 0.0, 4.0, 0.0, 4.0]


def test_out_of_order_appends_fail_closed():
    """Appends must be lexicographically increasing at runtime."""

    builder = _Builder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    x = builder.new_symbol_id()
    c = builder.new_symbol_id()
    decl_x = builder.tensor(
        x,
        "x",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        builder.dense_levels(2),
    )
    decl_c = builder.tensor(
        c,
        "C",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.DENSE, 0),
            builder.level(LevelKind.COMPRESSED, 1),
        ),
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    append = builder.append_entry(
        c,
        (builder.index_value(row), builder.index_value(col)),
        builder.load(x, (builder.index_value(row), builder.index_value(col))),
    )
    # The column loop is outermost, so appended rows regress between column
    # iterations; the oracle's order-checked builder must fail closed.
    program = builder.program(
        (dim_i, dim_j),
        (decl_x, decl_c),
        (x,),
        (c,),
        builder.block(
            (
                builder.dense_for(
                    col,
                    dim_j.dimension,
                    builder.block(
                        (
                            builder.dense_for(
                                row,
                                dim_i.dimension,
                                builder.block((append,)),
                            ),
                        )
                    ),
                ),
            )
        ),
    )
    with pytest.raises(LoopIROracleError):
        run_program(program, {x: [[1.0, 2.0], [3.0, 4.0]]}, {c: (2, 2)})


def test_sparse_input_binding_boundaries_fail_closed():
    fixture = build_csr_spmv()
    with pytest.raises(LoopIROracleError):
        run_program(
            fixture.program,
            {fixture.a: [[1.0, 0.0]], fixture.x: [1.0, 2.0]},
            {fixture.y: (1,)},
        )
    wrong_layout = LevelTensorStorage.from_dense(
        [[1.0, 0.0]], (1, 2), (0, 1), (LevelKind.COMPRESSED, LevelKind.COMPRESSED)
    )
    with pytest.raises(LoopIROracleError):
        run_program(
            fixture.program,
            {fixture.a: wrong_layout, fixture.x: [1.0, 2.0]},
            {fixture.y: (1,)},
        )


def test_sparse_binding_snapshots_detach_from_the_caller():
    fixture = build_csr_spmv()
    storage = LevelTensorStorage.from_dense(
        [[1.0, 0.0], [0.0, 2.0]],
        (2, 2),
        (0, 1),
        (LevelKind.DENSE, LevelKind.COMPRESSED),
    )
    results_before = run_program(
        fixture.program,
        {fixture.a: storage, fixture.x: [1.0, 1.0]},
        {fixture.y: (2,)},
    )
    fixture2 = build_csr_spmv()
    bound = LevelTensorStorage.from_dense(
        [[1.0, 0.0], [0.0, 2.0]],
        (2, 2),
        (0, 1),
        (LevelKind.DENSE, LevelKind.COMPRESSED),
    )
    # Mutating the caller's storage after binding cannot redirect execution
    # because the oracle snapshots the validated structure first.
    results = run_program(
        fixture2.program,
        {fixture2.a: bound, fixture2.x: [1.0, 1.0]},
        {fixture2.y: (2,)},
    )
    object.__setattr__(bound, "values", (9.0, 9.0))
    assert results[fixture2.y] == results_before[fixture.y]


def test_shared_dimension_extent_mismatch_between_storages_fails():
    fixture = build_union_add()
    a = csr_from_dense([[1.0, 0.0]])
    b = csr_from_dense([[1.0, 0.0, 2.0]])
    with pytest.raises(LoopIROracleError):
        run_program(
            fixture.program,
            {fixture.a: a, fixture.b: b},
            {fixture.c: (1, 2)},
        )
