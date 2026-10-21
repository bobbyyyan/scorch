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


def test_extent_mismatch_is_rejected_before_output_allocation(monkeypatch):
    import scorch.compiler.loopir.oracle as oracle_module

    fixture = build_vector_add()

    def unexpected_allocation(shape):
        raise AssertionError(f"allocated mismatched output shape {shape}")

    monkeypatch.setattr(oracle_module, "_zeros", unexpected_allocation)
    with pytest.raises(LoopIROracleError, match="dimension extent mismatch"):
        run_program(
            fixture.program,
            {fixture.a: [1.0], fixture.b: [2.0]},
            {fixture.c: (1_000_000,)},
        )


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


def test_sparse_extent_mismatch_is_rejected_before_builder_allocation(monkeypatch):
    import scorch.compiler.loopir.oracle as oracle_module

    fixture = build_union_add()

    def unexpected_builder(name, shape):
        raise AssertionError(f"allocated mismatched sparse output {name} {shape}")

    monkeypatch.setattr(oracle_module, "CsrOutputBuilder", unexpected_builder)
    with pytest.raises(LoopIROracleError, match="dimension extent mismatch"):
        run_program(
            fixture.program,
            {
                fixture.a: csr_from_dense([[1.0, 0.0]]),
                fixture.b: csr_from_dense([[0.0, 2.0]]),
            },
            {fixture.c: (1_000_000, 2)},
        )


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


def _one_stored_cell_into_compressed_over_dense():
    """``C[i, j] = A[j]`` for A's stored ``j``, with C compressed over dense."""

    builder = _Builder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    source = builder.new_symbol_id()
    output = builder.new_symbol_id()
    source_decl = builder.tensor(
        source,
        "A",
        ScalarType.FLOAT32,
        (dim_j.dimension,),
        (builder.level(LevelKind.COMPRESSED, 0),),
    )
    output_decl = builder.tensor(
        output,
        "C",
        ScalarType.FLOAT32,
        (dim_i.dimension, dim_j.dimension),
        (
            builder.level(LevelKind.COMPRESSED, 0),
            builder.level(LevelKind.DENSE, 1),
        ),
    )
    index_i = builder.new_index_id()
    index_j = builder.new_index_id()
    cursor = builder.new_cursor_id()
    position = builder.new_position_id()
    append = builder.append_entry(
        output,
        (builder.index_value(index_i), builder.index_value(index_j)),
        builder.cursor_value(cursor),
    )
    sparse = builder.sparse_for(
        builder.sparse_cursor(
            cursor,
            source,
            0,
            builder.root_position(),
        ),
        position,
        index_j,
        builder.block((append,)),
    )
    program = builder.program(
        (dim_i, dim_j),
        (source_decl, output_decl),
        (source,),
        (output,),
        builder.block(
            (
                builder.dense_for(
                    index_i,
                    dim_i.dimension,
                    builder.block((sparse,)),
                ),
            )
        ),
    )
    return program, source, output


def test_oracle_zero_fills_the_cells_of_a_dense_block_no_append_addressed():
    """A stored dense block holds every cell; unappended ones hold zero.

    The stream stores ``j = 0`` only, so the block under ``i = 0`` receives one
    append.  The coordinate is stored because something was appended under it,
    and the block's other cell is a position holding zero -- the storage rule
    the compiled kernels follow, not an incomplete assembly.
    """

    program, source, output = _one_stored_cell_into_compressed_over_dense()
    storage = LevelTensorStorage.from_dense(
        [1.0, 0.0],
        (2,),
        (0,),
        (LevelKind.COMPRESSED,),
    )
    result = run_program(program, {source: storage}, {output: (1, 2)})[output]
    assert result.positions[0] == (0, 1)
    assert result.coordinates[0] == (0,)
    assert result.values == (1.0, 0.0)


def test_oracle_wraps_level_assembly_failures():
    """A block too large to hold is reported as the output's assembly failure."""

    from scorch.compiler.loopir.levels import CompressedLevel

    program, source, output = _one_stored_cell_into_compressed_over_dense()
    huge = 10**30
    storage = LevelTensorStorage(
        shape=(huge,),
        modes=(0,),
        levels=(CompressedLevel((0, 1), (0,)),),
        values=(1.0,),
    )
    with pytest.raises(LoopIROracleError, match="output C assembly failed"):
        run_program(program, {source: storage}, {output: (1, huge)})


def test_oracle_diagnostics_bound_huge_exact_extents():
    fixture = build_vector_add()
    huge = 10**5000
    with pytest.raises(LoopIROracleError, match="integer too large to render"):
        run_program(
            fixture.program,
            {fixture.a: [1.0], fixture.b: [2.0]},
            {fixture.c: (huge,)},
        )


def test_out_of_order_appends_fail_closed():
    """Appends must be lexicographically increasing.

    The verifier refuses an append out of the result's storage order unless a
    two-pass record and a relabelled copy of one input admit it; the oracle's
    order-checked builder fails closed on the same program run unverified.
    """
    from scorch.compiler.loopir.verifier import LoopIRVerificationError

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
    with pytest.raises(LoopIRVerificationError) as refused:
        run_program(program, {x: [[1.0, 2.0], [3.0, 4.0]]}, {c: (2, 2)})
    assert refused.value.defect.code == "unordered_append"
    with pytest.raises(LoopIROracleError, match="lexicographically increasing"):
        _execute_unverified(program, {x: [[1.0, 2.0], [3.0, 4.0]]}, {c: (2, 2)})


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


# -- Phase-6 affine-split execution -------------------------------------------


def test_tiled_matvec_matches_the_unsplit_program_exactly():
    from tests.test_scorch.test_loopir_verifier import build_tiled_matvec

    rng = random.Random(626)
    for width in (1, 2, 3, 4, 5, 9):
        fixture = build_tiled_matvec(width=width)
        rows, cols = 3, 5
        matrix = [
            [float(rng.randrange(-3, 4)) for _ in range(cols)] for _ in range(rows)
        ]
        vector = [float(rng.randrange(-3, 4)) for _ in range(cols)]
        a, x = fixture.program.inputs
        (y,) = fixture.program.outputs
        results = run_program(fixture.program, {a: matrix, x: vector}, {y: (rows,)})
        assert results[y] == reference_matvec(matrix, vector)


def test_tiled_zero_extent_executes_nothing():
    from tests.test_scorch.test_loopir_verifier import build_tiled_matvec

    fixture = build_tiled_matvec(width=4)
    a, x = fixture.program.inputs
    (y,) = fixture.program.outputs
    results = run_program(fixture.program, {a: [[], [], []], x: []}, {y: (3,)})
    assert results[y] == [0.0, 0.0, 0.0]


def test_tiled_visits_every_coordinate_exactly_once():
    from tests.test_scorch.test_loopir_verifier import build_tiled_matvec

    for width, cols in ((1, 5), (2, 5), (5, 5), (7, 5)):
        fixture = build_tiled_matvec(width=width)
        rows = 2
        ones = [[1.0] * cols for _ in range(rows)]
        a, x = fixture.program.inputs
        (y,) = fixture.program.outputs
        results = run_program(fixture.program, {a: ones, x: [1.0] * cols}, {y: (rows,)})
        assert results[y] == [float(cols)] * rows


def test_point_loop_outside_its_origin_fails_closed_at_runtime():
    """Defensive boundary: the verifier's scope rule normally precludes this,
    so the oracle's runtime guard is exercised on the raw execution state."""

    from scorch.compiler.loopir.oracle import _Oracle
    from tests.test_scorch.test_loopir_verifier import build_tiled_matvec

    fixture = build_tiled_matvec(width=4)
    a, x = fixture.program.inputs
    (y,) = fixture.program.outputs
    oracle = _Oracle(
        fixture.program,
        {a: [[1.0, 2.0]], x: [1.0, 1.0]},
        {y: (1,)},
    )
    inner = fixture.inner
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(inner)
    assert "outside its origin loop" in str(error.value)


# -- Phase-6 workspace regions ------------------------------------------------


def stack_reference_matmul(a, b, n):
    rows, inner = len(a), len(b)
    return [
        [sum(a[i][j] * b[j][k] for j in range(inner)) for k in range(n)]
        for i in range(rows)
    ]


def test_workspace_region_matches_the_direct_program_exactly():
    from tests.test_scorch.test_loopir_verifier import build_stack_matmul

    rng = random.Random(627)
    for width in (1, 2, 3, 4, 5, 9):
        fixture = build_stack_matmul(width=width)
        rows, inner, cols = 3, 4, 5
        matrix_a = [
            [float(rng.randrange(-3, 4)) for _ in range(inner)] for _ in range(rows)
        ]
        matrix_b = [
            [float(rng.randrange(-3, 4)) for _ in range(cols)] for _ in range(inner)
        ]
        results = run_program(
            fixture.program,
            {fixture.a: matrix_a, fixture.b: matrix_b},
            {fixture.c: (rows, cols)},
        )
        assert results[fixture.c] == stack_reference_matmul(matrix_a, matrix_b, cols)


def test_workspace_region_resets_per_tile_and_per_row():
    """A stale workspace would double-count: with all-ones counting inputs
    every output cell must count exactly the reduction extent, for every
    row and every tile of the ragged split."""

    from tests.test_scorch.test_loopir_verifier import build_stack_matmul

    for width, cols in ((1, 5), (2, 5), (4, 5), (5, 5), (9, 5)):
        fixture = build_stack_matmul(width=width)
        rows, inner = 3, 4
        ones_a = [[1.0] * inner for _ in range(rows)]
        ones_b = [[1.0] * cols for _ in range(inner)]
        results = run_program(
            fixture.program,
            {fixture.a: ones_a, fixture.b: ones_b},
            {fixture.c: (rows, cols)},
        )
        assert results[fixture.c] == [[float(inner)] * cols for _ in range(rows)]


def test_workspace_region_does_not_eagerly_allocate_its_semantic_extent():
    """A huge target-neutral width still touches only executed point cells."""

    from tests.test_scorch.test_loopir_verifier import build_stack_matmul

    fixture = build_stack_matmul(width=1 << 100)
    results = run_program(
        fixture.program,
        {fixture.a: [[2.0]], fixture.b: [[3.0]]},
        {fixture.c: (1, 1)},
    )
    assert results[fixture.c] == [[6.0]]


def test_workspace_region_zero_extents_execute():
    from tests.test_scorch.test_loopir_verifier import build_stack_matmul

    fixture = build_stack_matmul(width=4)
    results = run_program(
        fixture.program,
        {fixture.a: [[], [], []], fixture.b: []},
        {fixture.c: (3, 0)},
    )
    assert results[fixture.c] == [[], [], []]
    fixture = build_stack_matmul(width=4)
    results = run_program(
        fixture.program,
        {fixture.a: [], fixture.b: [[1.0, 2.0]]},
        {fixture.c: (0, 2)},
    )
    assert results[fixture.c] == []


def test_workspace_access_outside_region_fails_closed_at_runtime():
    """Defensive boundary: the verifier's region scope rule normally
    precludes this, so the oracle's runtime guard is exercised directly."""

    from scorch.compiler.loopir.oracle import _Oracle
    from tests.test_scorch.test_loopir_verifier import build_stack_matmul

    fixture = build_stack_matmul(width=4)
    oracle = _Oracle(
        fixture.program,
        {fixture.a: [[1.0]], fixture.b: [[1.0]]},
        {fixture.c: (1, 1)},
    )
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(fixture.reduce_stmt)
    assert "outside its region" in str(error.value)
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(fixture.region)
    assert "outside its tile's origin loop" in str(error.value)


# -- Phase-6 sparse coordinate panels -----------------------------------------


def panel_reference_spmm(a_dense, b_dense, cols):
    rows = len(a_dense)
    inner = len(b_dense)
    return [
        [sum(a_dense[i][j] * b_dense[j][k] for j in range(inner)) for k in range(cols)]
        for i in range(rows)
    ]


def test_panel_spmm_matches_the_unwindowed_reference_exactly():
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    rng = random.Random(628)
    rows, inner, cols = 4, 5, 3
    for width in (1, 2, 3, 5, 7, 64):
        fixture = build_panel_spmm(width=width)
        a_dense = [
            [
                float(rng.randrange(-3, 4)) if rng.random() < 0.6 else 0.0
                for _ in range(inner)
            ]
            for _ in range(rows)
        ]
        b_dense = [
            [float(rng.randrange(-3, 4)) for _ in range(cols)] for _ in range(inner)
        ]
        results = run_program(
            fixture.program,
            {fixture.a: csr_from_dense(a_dense), fixture.b: b_dense},
            {fixture.c: (rows, cols)},
        )
        assert results[fixture.c] == panel_reference_spmm(a_dense, b_dense, cols)


def test_panel_visits_each_stored_entry_exactly_once():
    """Counting differential: with all-ones inputs each output cell must
    count exactly its row's stored entries, across ragged windows, empty
    rows, and disjoint per-row supports."""

    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    inner, cols = 7, 2
    a_dense = [
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0] * inner,
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0] * inner,
    ]
    stored_counts = [4.0, 0.0, 3.0, 7.0]
    for width in (1, 2, 3, 4, 7, 9):
        fixture = build_panel_spmm(width=width)
        ones_b = [[1.0] * cols for _ in range(inner)]
        results = run_program(
            fixture.program,
            {fixture.a: csr_from_dense(a_dense), fixture.b: ones_b},
            {fixture.c: (len(a_dense), cols)},
        )
        assert results[fixture.c] == [[count] * cols for count in stored_counts]


def test_panel_zero_extents_execute():
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    # Zero panel dimension: the origin loop runs no windows.
    fixture = build_panel_spmm(width=4)
    empty_cols = CsrMatrix(
        n_rows=3, n_cols=0, indptr=(0, 0, 0, 0), indices=(), values=()
    )
    results = run_program(
        fixture.program,
        {fixture.a: empty_cols, fixture.b: []},
        {fixture.c: (3, 2)},
    )
    assert results[fixture.c] == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    # Zero rows: the row loop under the panel runs nothing.
    fixture = build_panel_spmm(width=4)
    no_rows = CsrMatrix(n_rows=0, n_cols=2, indptr=(0,), indices=(), values=())
    results = run_program(
        fixture.program,
        {fixture.a: no_rows, fixture.b: [[1.0], [1.0]]},
        {fixture.c: (0, 1)},
    )
    assert results[fixture.c] == []


def test_panel_width_is_semantic_not_an_allocation_request():
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    fixture = build_panel_spmm(width=1 << 100)
    results = run_program(
        fixture.program,
        {fixture.a: csr_from_dense([[2.0]]), fixture.b: [[3.0]]},
        {fixture.c: (1, 1)},
    )
    assert results[fixture.c] == [[6.0]]


def test_window_outside_its_panel_fails_closed_at_runtime():
    """Defensive boundary: the verifier's panel scope rule normally
    precludes this, so the oracle's runtime guard is exercised directly."""

    from scorch.compiler.loopir.oracle import _Oracle
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    fixture = build_panel_spmm(width=4)
    oracle = _Oracle(
        fixture.program,
        {fixture.a: csr_from_dense([[1.0]]), fixture.b: [[1.0]]},
        {fixture.c: (1, 1)},
    )
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(fixture.row_loop)
    assert "outside its panel's origin loop" in str(error.value)


def test_relayout_spmm_matches_the_unstaged_reference_exactly():
    from scorch.compiler.loopir.nodes import RelayoutScope

    from tests.test_scorch.test_loopir_verifier import build_relayout_spmm

    rng = random.Random(1729)
    rows, inner, cols = 4, 5, 6
    for scope in (RelayoutScope.PANEL, RelayoutScope.PACK_AXIS):
        for width, strip in ((1, 1), (2, 3), (3, 4), (5, 6), (7, 9)):
            fixture = build_relayout_spmm(scope, width=width, strip=strip)
            a_dense = [
                [
                    float(rng.randrange(-3, 4)) if rng.random() < 0.6 else 0.0
                    for _ in range(inner)
                ]
                for _ in range(rows)
            ]
            b_dense = [
                [float(rng.randrange(-3, 4)) for _ in range(cols)] for _ in range(inner)
            ]
            results = run_program(
                fixture.program,
                {fixture.a: csr_from_dense(a_dense), fixture.b: b_dense},
                {fixture.c: (rows, cols)},
            )
            assert results[fixture.c] == panel_reference_spmm(a_dense, b_dense, cols)


def test_relayout_counts_each_stored_entry_and_column_exactly_once():
    """Counting differential: all-ones inputs count row nnz per output cell
    across ragged panel windows and ragged pack strips in both scopes.

    The oracle serves immutable staged inputs lazily, so this locks compute
    visitation rather than physical copy frequency; compiled target tests
    own staging-placement and freshness evidence.
    """

    from scorch.compiler.loopir.nodes import RelayoutScope

    from tests.test_scorch.test_loopir_verifier import build_relayout_spmm

    inner, cols = 7, 5
    a_dense = [
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0] * inner,
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0] * inner,
    ]
    stored_counts = [4.0, 0.0, 3.0, 7.0]
    ones_b = [[1.0] * cols for _ in range(inner)]
    for scope in (RelayoutScope.PANEL, RelayoutScope.PACK_AXIS):
        for width, strip in ((1, 2), (2, 5), (3, 3), (9, 8)):
            fixture = build_relayout_spmm(scope, width=width, strip=strip)
            results = run_program(
                fixture.program,
                {fixture.a: csr_from_dense(a_dense), fixture.b: ones_b},
                {fixture.c: (len(a_dense), cols)},
            )
            assert results[fixture.c] == [[count] * cols for count in stored_counts]


def test_relayout_zero_extents_execute():
    from scorch.compiler.loopir.nodes import RelayoutScope

    for scope in (RelayoutScope.PANEL, RelayoutScope.PACK_AXIS):
        from tests.test_scorch.test_loopir_verifier import build_relayout_spmm

        # Zero panel dimension: no windows, no staged strips.
        fixture = build_relayout_spmm(scope, width=4, strip=2)
        empty_cols = CsrMatrix(
            n_rows=3, n_cols=0, indptr=(0, 0, 0, 0), indices=(), values=()
        )
        results = run_program(
            fixture.program,
            {fixture.a: empty_cols, fixture.b: []},
            {fixture.c: (3, 2)},
        )
        assert results[fixture.c] == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

        # Zero rows.
        fixture = build_relayout_spmm(scope, width=4, strip=2)
        no_rows = CsrMatrix(n_rows=0, n_cols=2, indptr=(0,), indices=(), values=())
        results = run_program(
            fixture.program,
            {fixture.a: no_rows, fixture.b: [[1.0, 2.0], [3.0, 4.0]]},
            {fixture.c: (0, 2)},
        )
        assert results[fixture.c] == []

        # Zero free (pack) dimension: the pack origin loop runs nothing.
        fixture = build_relayout_spmm(scope, width=2, strip=3)
        results = run_program(
            fixture.program,
            {fixture.a: csr_from_dense([[1.0]]), fixture.b: [[]]},
            {fixture.c: (1, 0)},
        )
        assert results[fixture.c] == [[]]


def test_relayout_widths_are_semantic_not_allocation_requests():
    from tests.test_scorch.test_loopir_verifier import build_relayout_spmm

    fixture = build_relayout_spmm(width=1 << 100, strip=1 << 90)
    results = run_program(
        fixture.program,
        {fixture.a: csr_from_dense([[2.0]]), fixture.b: [[3.0]]},
        {fixture.c: (1, 1)},
    )
    assert results[fixture.c] == [[6.0]]


def test_staged_read_outside_its_region_fails_closed_at_runtime():
    """Defensive boundary: the verifier normally precludes these, so the
    oracle's runtime guards are exercised directly."""

    from scorch.compiler.loopir.nodes import RelayoutScope
    from scorch.compiler.loopir.oracle import _Oracle

    from tests.test_scorch.test_loopir_verifier import build_relayout_spmm

    fixture = build_relayout_spmm(RelayoutScope.PANEL, width=2, strip=2)
    oracle = _Oracle(
        fixture.program,
        {fixture.a: csr_from_dense([[1.0]]), fixture.b: [[1.0]]},
        {fixture.c: (1, 1)},
    )
    with pytest.raises(LoopIROracleError) as error:
        oracle._eval(fixture.staged)
    assert "outside its relayout region's execution" in str(error.value)

    # A region executed outside its pack origin fails closed.
    oracle = _Oracle(
        fixture.program,
        {fixture.a: csr_from_dense([[1.0]]), fixture.b: [[1.0]]},
        {fixture.c: (1, 1)},
    )
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(fixture.panel)
    assert "outside its pack split's origin loop" in str(error.value)

    # A PANEL-scoped region outside its panel origin fails closed.
    oracle = _Oracle(
        fixture.program,
        {fixture.a: csr_from_dense([[1.0]]), fixture.b: [[1.0]]},
        {fixture.c: (1, 1)},
    )
    oracle.tile_origins[fixture.pack_tile] = 0
    oracle._tile_widths[fixture.pack_tile] = 2
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(fixture.stage)
    assert "outside its panel's origin loop" in str(error.value)

    # Re-entering an executing region fails closed.
    oracle = _Oracle(
        fixture.program,
        {fixture.a: csr_from_dense([[1.0]]), fixture.b: [[1.0]]},
        {fixture.c: (1, 1)},
    )
    oracle.tile_origins[fixture.pack_tile] = 0
    oracle._tile_widths[fixture.pack_tile] = 2
    oracle.panel_origins[fixture.panel_tile] = 0
    oracle._panel_widths[fixture.panel_tile] = 2
    oracle.staged_relayouts[fixture.relayout] = fixture.decl
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(fixture.stage)
    assert "re-entered" in str(error.value)


def test_staged_read_outside_the_staged_domain_fails_closed_at_runtime():
    from scorch.compiler.loopir.nodes import RelayoutScope
    from scorch.compiler.loopir.oracle import _Oracle

    from tests.test_scorch.test_loopir_verifier import build_relayout_spmm

    # Column outside the current pack window.
    fixture = build_relayout_spmm(RelayoutScope.PANEL, width=4, strip=2)
    oracle = _Oracle(
        fixture.program,
        {
            fixture.a: csr_from_dense([[1.0, 1.0, 1.0, 1.0]]),
            fixture.b: [[1.0] * 6 for _ in range(4)],
        },
        {fixture.c: (1, 6)},
    )
    oracle.staged_relayouts[fixture.relayout] = fixture.decl
    oracle.tile_origins[fixture.pack_tile] = 0
    oracle._tile_widths[fixture.pack_tile] = 2
    oracle.panel_origins[fixture.panel_tile] = 0
    oracle._panel_widths[fixture.panel_tile] = 4
    oracle.indices[fixture.col] = 0
    oracle.indices[fixture.free] = 3  # outside pack window [0, 2)
    with pytest.raises(LoopIROracleError) as error:
        oracle._eval(fixture.staged)
    assert "outside the current pack window" in str(error.value)

    # Row outside the current panel window (PANEL scope).
    oracle.indices[fixture.free] = 1
    oracle.panel_origins[fixture.panel_tile] = 0
    oracle._panel_widths[fixture.panel_tile] = 1
    oracle.indices[fixture.col] = 2  # outside panel window [0, 1)
    with pytest.raises(LoopIROracleError) as error:
        oracle._eval(fixture.staged)
    assert "outside the current panel window" in str(error.value)


def test_heap_result_tile_matches_the_direct_reference_exactly():
    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    rng = random.Random(2027)
    rows, inner, cols = 4, 5, 6
    for strip in (1, 2, 3, 4, 6, 9):
        fixture = build_heap_spmm(strip=strip)
        a_dense = [
            [
                float(rng.randrange(-3, 4)) if rng.random() < 0.6 else 0.0
                for _ in range(inner)
            ]
            for _ in range(rows)
        ]
        b_dense = [
            [float(rng.randrange(-3, 4)) for _ in range(cols)] for _ in range(inner)
        ]
        results = run_program(
            fixture.program,
            {fixture.a: csr_from_dense(a_dense), fixture.b: b_dense},
            {fixture.c: (rows, cols)},
        )
        assert results[fixture.c] == panel_reference_spmm(a_dense, b_dense, cols)


def test_heap_result_tile_copies_out_fresh_strips_exactly_once():
    """Freshness/coverage proof: every strip is reset and every cell copied.

    The oracle's duplicate-key guard separately enforces at-most-once
    copy-out within one region execution; repeated assignment itself would
    be idempotent and therefore is not inferred from these numeric values.
    """

    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    inner, cols = 7, 5
    a_dense = [
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0] * inner,
        [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        [1.0] * inner,
    ]
    stored_counts = [4.0, 0.0, 3.0, 7.0]
    ones_b = [[1.0] * cols for _ in range(inner)]
    for strip in (1, 2, 3, 5, 8):
        fixture = build_heap_spmm(strip=strip)
        results = run_program(
            fixture.program,
            {fixture.a: csr_from_dense(a_dense), fixture.b: ones_b},
            {fixture.c: (len(a_dense), cols)},
        )
        assert results[fixture.c] == [[count] * cols for count in stored_counts]


def test_heap_result_tile_copy_out_respects_physical_mode_permutations():
    """The compact axis is the final physical level, not logical rank-1."""

    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    fixture = build_heap_spmm(strip=2)
    result_decl = fixture.program.tensors[2]
    forge(
        result_decl,
        dimensions=(fixture.dim_k, fixture.dim_i),
        levels=(
            fixture.builder.level(
                result_decl.levels[0].kind,
                1,
            ),
            fixture.builder.level(
                result_decl.levels[1].kind,
                0,
            ),
        ),
    )
    forge(
        fixture.leaf,
        indices=(
            fixture.builder.index_value(fixture.free),
            fixture.builder.index_value(fixture.row),
        ),
    )
    verify_program(fixture.program)

    a_dense = [[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]]
    b_dense = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    ordinary = panel_reference_spmm(a_dense, b_dense, 2)
    results = run_program(
        fixture.program,
        {fixture.a: csr_from_dense(a_dense), fixture.b: b_dense},
        {fixture.c: (2, 2)},
    )
    assert results[fixture.c] == [list(column) for column in zip(*ordinary)]


def test_heap_result_tile_zero_extents_execute():
    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    # Zero reduction dimension: stored entries never fire, copy-out still
    # writes zeros across every strip.
    fixture = build_heap_spmm(strip=2)
    empty_cols = CsrMatrix(
        n_rows=3, n_cols=0, indptr=(0, 0, 0, 0), indices=(), values=()
    )
    results = run_program(
        fixture.program,
        {fixture.a: empty_cols, fixture.b: []},
        {fixture.c: (3, 2)},
    )
    assert results[fixture.c] == [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

    # Zero rows: an empty prefix space; copy-out enumerates nothing.
    fixture = build_heap_spmm(strip=2)
    no_rows = CsrMatrix(n_rows=0, n_cols=2, indptr=(0,), indices=(), values=())
    results = run_program(
        fixture.program,
        {fixture.a: no_rows, fixture.b: [[1.0, 2.0], [3.0, 4.0]]},
        {fixture.c: (0, 2)},
    )
    assert results[fixture.c] == []

    # Zero free (pack) dimension: the pack origin loop runs nothing.
    fixture = build_heap_spmm(strip=3)
    results = run_program(
        fixture.program,
        {fixture.a: csr_from_dense([[1.0]]), fixture.b: [[]]},
        {fixture.c: (1, 0)},
    )
    assert results[fixture.c] == [[]]


def test_heap_result_tile_widths_are_semantic_not_allocation_requests():
    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    fixture = build_heap_spmm(strip=1 << 100)
    results = run_program(
        fixture.program,
        {fixture.a: csr_from_dense([[2.0]]), fixture.b: [[3.0]]},
        {fixture.c: (1, 1)},
    )
    assert results[fixture.c] == [[6.0]]


def test_tiled_reduce_outside_its_region_fails_closed_at_runtime():
    """Defensive boundary: the verifier normally precludes these, so the
    oracle's runtime guards are exercised directly."""

    from scorch.compiler.loopir.oracle import _Oracle

    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    fixture = build_heap_spmm(strip=2)
    oracle = _Oracle(
        fixture.program,
        {fixture.a: csr_from_dense([[1.0]]), fixture.b: [[1.0]]},
        {fixture.c: (1, 1)},
    )
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(fixture.leaf)
    assert "outside its result-tile region's execution" in str(error.value)

    # The region itself outside its pack origin fails closed.
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(fixture.region)
    assert "outside its pack split's origin loop" in str(error.value)

    # Re-entering the live region fails closed.
    oracle.tile_origins[fixture.pack_tile] = 0
    oracle._tile_widths[fixture.pack_tile] = 2
    oracle.result_tiles[fixture.result_tile] = (fixture.decl, {})
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(fixture.region)
    assert "re-entered" in str(error.value)


def test_tiled_reduce_outside_the_compact_domain_fails_closed_at_runtime():
    from scorch.compiler.loopir.oracle import _Oracle

    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    fixture = build_heap_spmm(strip=2)
    oracle = _Oracle(
        fixture.program,
        {
            fixture.a: csr_from_dense([[1.0, 1.0, 1.0, 1.0]]),
            fixture.b: [[1.0] * 6 for _ in range(4)],
        },
        {fixture.c: (1, 6)},
    )
    oracle.result_tiles[fixture.result_tile] = (fixture.decl, {})
    oracle.tile_origins[fixture.pack_tile] = 0
    oracle._tile_widths[fixture.pack_tile] = 2
    oracle.indices[fixture.row] = 0
    oracle.indices[fixture.col] = 0
    oracle.positions.clear()
    oracle.indices[fixture.free] = 3  # outside pack window [0, 2)
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_tiled_reduce(fixture.leaf)
    assert "outside the current pack window" in str(error.value)

    # A prefix coordinate outside the result's dimension extent.
    oracle.indices[fixture.free] = 1
    oracle.indices[fixture.row] = 5
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_tiled_reduce(fixture.leaf)
    assert "out of bounds" in str(error.value)


# -- Phase-7 position loads through a dense leaf below compressed --------------


def test_position_load_copy_matches_the_dense_reference():
    from tests.test_scorch.test_loopir_verifier import (
        build_mixed_leaf_operand_copy,
    )

    fixture = build_mixed_leaf_operand_copy()
    dense = [
        [1.0, 0.0, 2.0],
        [0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    storage = LevelTensorStorage.from_dense(
        dense, (4, 3), (0, 1), (LevelKind.COMPRESSED, LevelKind.DENSE)
    )
    results = run_program(fixture.program, {fixture.a: storage}, {fixture.c: (4, 3)})
    assert results[fixture.c] == dense


def test_position_load_preserves_stored_explicit_zeros_per_row():
    """A stored row's dense leaf serves explicit zeros, not pruned holes."""

    from scorch.compiler.loopir.levels import CompressedLevel, DenseLevel

    from tests.test_scorch.test_loopir_verifier import (
        build_mixed_leaf_operand_copy,
    )

    fixture = build_mixed_leaf_operand_copy()
    storage = LevelTensorStorage(
        shape=(3, 2),
        modes=(0, 1),
        levels=(
            CompressedLevel((0, 1), (1,)),
            DenseLevel(2),
        ),
        values=(0.0, 7.0),
    )
    results = run_program(fixture.program, {fixture.a: storage}, {fixture.c: (3, 2)})
    assert results[fixture.c] == [[0.0, 0.0], [0.0, 7.0], [0.0, 0.0]]


def test_position_load_materializes_an_all_dense_input_lazily():
    """A hand-built position load over an all-dense input still executes."""

    builder = _Builder()
    dim_i = builder.dimension("i")
    a, c = (builder.new_symbol_id() for _ in range(2))
    decl_a = builder.tensor(
        a, "A", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    decl_c = builder.tensor(
        c, "C", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    index = builder.new_index_id()
    load = builder.position_load(
        a,
        builder.dense_position(
            a, 0, builder.root_position(), builder.index_value(index)
        ),
    )
    leaf = builder.store(c, (builder.index_value(index),), load)
    loop = builder.dense_for(index, dim_i.dimension, builder.block((leaf,)))
    program = builder.program(
        (dim_i,), (decl_a, decl_c), (a,), (c,), builder.block((loop,))
    )
    results = run_program(program, {a: [4.0, 0.0, -1.5]}, {c: (3,)})
    assert results[c] == [4.0, 0.0, -1.5]


def test_position_load_outside_leaf_storage_fails_closed_at_runtime():
    """A malformed runtime position is an oracle error, not a wrong value."""

    from scorch.compiler.loopir.levels import CompressedLevel, DenseLevel
    from scorch.compiler.loopir.oracle import _Oracle

    from tests.test_scorch.test_loopir_verifier import (
        build_mixed_leaf_operand_copy,
    )

    fixture = build_mixed_leaf_operand_copy()
    storage = LevelTensorStorage(
        shape=(4, 3),
        modes=(0, 1),
        levels=(
            CompressedLevel((0, 2), (0, 2)),
            DenseLevel(3),
        ),
        values=(1.0, 0.0, 2.0, 0.0, 3.0, 0.0),
    )
    oracle = _Oracle(fixture.program, {fixture.a: storage}, {fixture.c: (4, 3)})
    oracle.positions[fixture.program.body.statements[0].position] = 5
    oracle.indices[fixture.col] = 0
    oracle.indices[fixture.row] = 0
    with pytest.raises(LoopIROracleError) as error:
        oracle._eval(fixture.load)
    assert "leaf storage" in str(error.value)


@pytest.mark.parametrize("callback", ["iter", "getitem"])
@pytest.mark.parametrize("mutated_mapping", ["inputs", "outputs"])
def test_binding_mapping_cannot_mutate_position_load_after_verification(
    mutated_mapping, callback
):
    """Caller mapping callbacks run before the oracle trusts the program."""

    from scorch.compiler.loopir.verifier import LoopIRVerificationError

    from tests.test_scorch.test_loopir_verifier import (
        build_mixed_leaf_operand_copy,
    )

    fixture = build_mixed_leaf_operand_copy()
    storage = LevelTensorStorage.from_dense(
        [[1.0, 0.0], [0.0, 2.0]],
        (2, 2),
        (0, 1),
        (LevelKind.COMPRESSED, LevelKind.DENSE),
    )

    class MutatingMapping(dict):
        def __iter__(self):
            if callback == "iter":
                object.__setattr__(fixture.load, "tensor", [])
            return super().__iter__()

        def __getitem__(self, key):
            if callback == "getitem":
                object.__setattr__(fixture.load, "tensor", [])
            return super().__getitem__(key)

    inputs = {fixture.a: storage}
    outputs = {fixture.c: (2, 2)}
    if mutated_mapping == "inputs":
        inputs = MutatingMapping(inputs)
    else:
        outputs = MutatingMapping(outputs)

    with pytest.raises(LoopIRVerificationError) as error:
        run_program(fixture.program, inputs, outputs)
    assert error.value.defect.code == "invalid_symbol_id"


def test_malformed_position_load_fails_before_binding_mapping_callbacks():
    """The initial verifier remains cheaper than caller-controlled mappings."""

    from scorch.compiler.loopir.verifier import LoopIRVerificationError

    from tests.test_scorch.test_loopir_verifier import (
        build_mixed_leaf_operand_copy,
    )

    fixture = build_mixed_leaf_operand_copy()
    object.__setattr__(fixture.load, "tensor", [])
    called = False

    class HostileBindings(dict):
        def __iter__(self):
            nonlocal called
            called = True
            raise RuntimeError("binding mapping must not run")

    with pytest.raises(LoopIRVerificationError) as error:
        run_program(
            fixture.program,
            HostileBindings({fixture.a: object()}),
            {fixture.c: (2, 2)},
        )
    assert error.value.defect.code == "invalid_symbol_id"
    assert not called


def test_foreign_sparse_binding_class_is_rejected_without_invocation():
    """A hostile compressed binding fails the exact-type check uninvoked."""

    from tests.test_scorch.test_loopir_verifier import (
        build_mixed_leaf_operand_copy,
    )

    fixture = build_mixed_leaf_operand_copy()
    invoked = []

    class HostileStorage:
        def __getattr__(self, name):
            invoked.append(name)
            raise RuntimeError("must never be inspected")

    with pytest.raises(LoopIROracleError) as error:
        run_program(
            fixture.program,
            {fixture.a: HostileStorage()},
            {fixture.c: (2, 2)},
        )
    assert "LevelTensorStorage" in str(error.value)
    assert invoked == []


def test_nested_dense_binding_ownership_precedes_output_shape_lookup():
    """Deep dense containers are owned before output-shape callbacks run."""

    fixture = build_matmul()
    inner = [1.0, 2.0]
    a = [inner, [3.0, 4.0]]
    b = [[1.0, 0.0], [0.0, 1.0]]

    class InnerMutatingShapes(dict):
        def __getitem__(self, key):
            inner[0] = 999.0
            return super().__getitem__(key)

    results = run_program(
        fixture.program,
        {fixture.a: a, fixture.b: b},
        InnerMutatingShapes({fixture.c: (2, 2)}),
    )
    assert results[fixture.c] == [[1.0, 2.0], [3.0, 4.0]]
    assert inner[0] == 999.0


def test_oracle_binding_key_mutation_during_lookup_fails_closed():
    """Mutating a SymbolId key inside ``__getitem__`` cannot pass snapshot."""

    from scorch.compiler.loopir.levels import LevelTensorStorage as _Storage
    from scorch.compiler.loopir.nodes import LevelKind as _Kind

    from tests.test_scorch.test_loopir_verifier import (
        build_mixed_leaf_operand_copy,
    )

    fixture = build_mixed_leaf_operand_copy()
    storage = _Storage.from_dense(
        [[1.0, 0.0], [0.0, 2.0]],
        (2, 2),
        (0, 1),
        (_Kind.COMPRESSED, _Kind.DENSE),
    )

    class KeyMutating(dict):
        def __getitem__(self, key):
            value = super().__getitem__(key)
            object.__setattr__(key, "value", key.value + 1000)
            return value

    with pytest.raises(LoopIROracleError) as error:
        run_program(
            fixture.program,
            KeyMutating({fixture.a: storage}),
            {fixture.c: (2, 2)},
        )
    assert "changed while values were being snapshotted" in str(error.value)


# -- rank-K sparse workspace keys -------------------------------------------
#
# The drain contract is strictly increasing LEXICOGRAPHIC key order.  When the
# key dimensions are listed in result level order that is also the canonical
# append order, which is what makes multi-level ordered result assembly
# correct without any sort in the IR.  These execute that contract directly.

from tests.test_scorch.test_loopir_verifier import (  # noqa: E402
    build_rank_k_workspace_program,
    rank_k_insert,
)


def _level_entries(tensor):
    """Expand hierarchical level storage into ``(key_tuple, value)`` pairs.

    Compressed levels are parent-linked, not a flat COO list: level ``L``'s
    coordinate array is indexed by position, and level ``L+1``'s position
    array segments it.  Zipping the coordinate arrays would be wrong -- the
    outer level stores one coordinate per distinct parent, not one per
    entry -- so this walks the position links, which is exactly the
    positions-are-not-coordinates discipline the assembly must honour.
    """

    entries = []

    def walk(level, parent_position, prefix):
        if level == len(tensor.level_kinds):
            entries.append((tuple(prefix), tensor.values[parent_position]))
            return
        positions = tensor.positions[level]
        coordinates = tensor.coordinates[level]
        for position in range(
            positions[parent_position], positions[parent_position + 1]
        ):
            walk(level + 1, position, prefix + [coordinates[position]])

    walk(0, 0, [])
    return entries


def _drained_keys(program, shape):
    output = program.outputs[0]
    results = run_program(program, {}, {output: shape})
    tensor = results[output]
    assert all(kind is LevelKind.COMPRESSED for kind in tensor.level_kinds)
    entries = _level_entries(tensor)
    return tensor, [key for key, _ in entries], [value for _, value in entries]


@pytest.mark.parametrize("key_rank", [1, 2, 3])
def test_rank_k_drain_visits_every_key_exactly_once(key_rank):
    """One entry per distinct key, merged under ADD, no duplicates."""

    shape = (2, 3, 2)[:key_rank]
    _, program, _, _ = build_rank_k_workspace_program(key_rank)
    verify_program(program)
    _, keys, values = _drained_keys(program, shape)

    expected = [()]
    for extent in shape:
        expected = [row + (value,) for row in expected for value in range(extent)]
    assert keys == expected
    assert len(keys) == len(set(keys))
    # Each key is inserted exactly once with 1.0 and ADD merges by key.
    assert values == [1.0] * len(expected)


def test_rank_2_drain_is_lexicographic_not_insertion_ordered():
    """Insertion order and key order deliberately disagree here.

    The producer's loop nest runs ``i`` then ``j`` while the key is
    ``(j, i)``, so insertions arrive ``(0,0), (1,0), (2,0), (0,1), ...``.
    A drain that merely preserved insertion order, or sorted by the leading
    component alone, would not produce the sequence below.  This is the
    case the inherited single-drain form could not express at all, and the
    ordering property multi-level ordered assembly rests on.
    """

    _, program, _, _ = build_rank_k_workspace_program(2, key_permutation=(1, 0))
    verify_program(program)
    _, keys, _ = _drained_keys(program, (3, 2))

    assert keys == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    assert keys == sorted(keys)


def test_rank_3_drain_is_lexicographic_under_a_rotated_key():
    """Rank 3 with the key rotated one step against the producer nest."""

    _, program, _, _ = build_rank_k_workspace_program(3, key_permutation=(2, 0, 1))
    verify_program(program)
    _, keys, _ = _drained_keys(program, (2, 2, 3))

    assert keys == sorted(keys)
    assert len(keys) == 12
    assert keys[0] == (0, 0, 0) and keys[-1] == (1, 1, 2)


@pytest.mark.parametrize("contraction_extent", [1, 4])
def test_rank_k_drain_merges_a_contraction_under_add(contraction_extent):
    """A contraction loop inside the region merges into one entry per key.

    The innermost producer loop's index does not appear in the key, so all
    ``contraction_extent`` of its iterations insert into the same entry.  The
    region must merge them under ADD and the drain must still visit each key
    exactly once -- the reduction shape this whole vertical exists for.  The
    merged values are checked against an independent Python reduction.
    """

    rows, columns = 2, 3
    _, program, _, _ = build_rank_k_workspace_program(
        2, contraction_extent=contraction_extent
    )
    verify_program(program)
    operand = [
        [
            [float(r * 100 + c * 10 + k) for k in range(contraction_extent)]
            for c in range(columns)
        ]
        for r in range(rows)
    ]
    output = program.outputs[0]
    results = run_program(
        program,
        {program.inputs[0]: operand},
        {output: (rows, columns)},
    )
    tensor = results[output]
    entries = _level_entries(tensor)
    keys = [key for key, _ in entries]
    values = [value for _, value in entries]

    assert keys == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert values == [sum(operand[r][c]) for r in range(rows) for c in range(columns)]


def test_rank_mismatched_insertion_fails_closed_in_the_oracle():
    """``zip`` would truncate to the shorter side; the executor must not.

    ``run_program`` verifies before executing, so the verifier is the
    authoritative boundary for this shape and a forged program never reaches
    the executor through it.  The executor still owns an explicit check
    rather than a silent truncation, driven here directly through the same
    ``_Oracle`` idiom the tile-scope fail-closed tests use.
    """

    from scorch.compiler.loopir.oracle import _Oracle

    _, program, region, _ = build_rank_k_workspace_program(2)
    insert = rank_k_insert(region)
    forge(insert, coords=insert.coords[:1])
    oracle = _Oracle(program, {}, {program.outputs[0]: (2, 3)})
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(program.body)
    assert "1 coordinates for a rank-2 key" in str(error.value)


def test_rank_mismatched_drain_fails_closed_in_the_oracle():
    from scorch.compiler.loopir.oracle import _Oracle

    _, program, _, drain = build_rank_k_workspace_program(2)
    forge(drain, indices=drain.indices[:1])
    oracle = _Oracle(program, {}, {program.outputs[0]: (2, 3)})
    with pytest.raises(LoopIROracleError) as error:
        oracle._exec_stmt(program.body)
    assert "binds 1 indices for a rank-2 key" in str(error.value)


# -- COO: tensors whose levels are all coordinate levels ----------------------
#
# The verifier's admission of COORDINATE levels is tested in its own file, so
# nothing here depends on it.  Programs with a coordinate level run on the
# executor directly, through the ``_Oracle`` idiom the fail-closed tests above
# use.  Each of them is also built over COMPRESSED levels, statement for
# statement.  Where that program has no coordinate level left, it also runs
# through ``run_program``, which verifies it.

import itertools  # noqa: E402

from scorch.compiler.loopir.levels import (  # noqa: E402
    CoordinateLevel,
    LevelTensor,
)

_COORDINATE = LevelKind.COORDINATE
_COMPRESSED = LevelKind.COMPRESSED


def _execute_unverified(program, inputs, output_shapes):
    from scorch.compiler.loopir.oracle import _Oracle

    return _Oracle(program, inputs, output_shapes).run()


def _stored_tensor_program(kind, modes, result, result_kind=_COORDINATE):
    """Walk one stored input level by level and write each entry once.

    Level ``l`` of the input ``A`` has ``kind`` and stores logical mode
    ``modes[l]``.  One ``sparse_for`` per level walks it; the cursor of
    level 0 takes the root position as its parent, and the cursor of every
    deeper level takes the position its enclosing loop binds.  ``result``
    selects the write:

    - ``"copy"`` stores the entry into a dense tensor of A's logical shape;
    - ``"sum"`` reduces the entry into a dense vector over logical mode 0;
    - ``"append"`` appends the entry to a result whose levels all have
      ``result_kind``, in logical mode order;
    - ``"reduce"`` reduces the entry into a rank-1 result of ``result_kind``
      over logical mode 0, the accumulation the oracle merges by coordinate.
    """

    rank = len(modes)
    builder = _Builder()
    dims = [builder.dimension(f"d{mode}") for mode in range(rank)]
    dimensions = tuple(dim.dimension for dim in dims)
    a = builder.new_symbol_id()
    out = builder.new_symbol_id()
    decl_a = builder.tensor(
        a,
        "A",
        ScalarType.FLOAT32,
        dimensions,
        tuple(builder.level(kind, mode) for mode in modes),
    )
    indices = [builder.new_index_id() for _ in range(rank)]
    cursors = [builder.new_cursor_id() for _ in range(rank)]
    positions = [builder.new_position_id() for _ in range(rank)]
    logical = tuple(builder.index_value(index) for index in indices)
    value = builder.cursor_value(cursors[-1])
    if result == "copy":
        decl_out = builder.tensor(
            out, "C", ScalarType.FLOAT32, dimensions, builder.dense_levels(rank)
        )
        leaf = builder.store(out, logical, value)
    elif result == "sum":
        decl_out = builder.tensor(
            out, "y", ScalarType.FLOAT32, dimensions[:1], builder.dense_levels(1)
        )
        leaf = builder.store_reduce(out, logical[:1], ReduceOp.ADD, value)
    elif result == "append":
        decl_out = builder.tensor(
            out,
            "C",
            ScalarType.FLOAT32,
            dimensions,
            tuple(builder.level(result_kind, mode) for mode in range(rank)),
        )
        leaf = builder.append_entry(out, logical, value)
    else:
        assert result == "reduce"
        decl_out = builder.tensor(
            out,
            "y",
            ScalarType.FLOAT32,
            dimensions[:1],
            (builder.level(result_kind, 0),),
        )
        leaf = builder.store_reduce(out, logical[:1], ReduceOp.ADD, value)
    body = leaf
    for level in reversed(range(rank)):
        parent = (
            builder.root_position()
            if level == 0
            else builder.position_value(positions[level - 1])
        )
        body = builder.sparse_for(
            builder.sparse_cursor(cursors[level], a, level, parent),
            positions[level],
            indices[modes[level]],
            builder.block((body,)),
        )
    program = builder.program(
        tuple(dims), (decl_a, decl_out), (a,), (out,), builder.block((body,))
    )
    return program, a, out


def _run_stored_tensor_program(
    kind, dense, shape, modes, result, result_kind=_COORDINATE
):
    """Run one stored-tensor program on ``dense`` stored with ``kind`` levels.

    Returns the result and the bound input storage.
    """

    rank = len(shape)
    program, a, out = _stored_tensor_program(kind, modes, result, result_kind)
    storage = LevelTensorStorage.from_dense(dense, shape, modes, (kind,) * rank)
    output_shape = shape if result in ("copy", "append") else shape[:1]
    produced = _execute_unverified(program, {a: storage}, {out: output_shape})[out]
    dense_result = result in ("copy", "sum")
    if kind is _COMPRESSED and (dense_result or result_kind is _COMPRESSED):
        # No level of this program is a coordinate level, so the verifier
        # admits it, and the verified entry point agrees with the executor.
        verified = run_program(program, {a: storage}, {out: output_shape})[out]
        assert verified == produced
    return produced, storage


def _nonzero_cells(dense, shape):
    """Logical coordinates and values of the nonzero cells, in sorted order."""

    cells = []
    for logical in itertools.product(*(range(extent) for extent in shape)):
        value = dense
        for coordinate in logical:
            value = value[coordinate]
        if value != 0.0:
            cells.append((logical, value))
    return cells


def _sum_over_mode_zero(dense, shape, modes):
    """Sums over every mode but mode 0, added in the storage order of ``modes``."""

    sums = [0.0] * shape[0]
    cells = sorted(
        (tuple(logical[mode] for mode in modes), logical, value)
        for logical, value in _nonzero_cells(dense, shape)
    )
    for _, logical, value in cells:
        sums[logical[0]] = sums[logical[0]] + value
    return sums


def _random_dense(rng, shape, density):
    if len(shape) == 1:
        return [
            float(rng.randint(1, 9)) if rng.random() < density else 0.0
            for _ in range(shape[0])
        ]
    return [_random_dense(rng, shape[1:], density) for _ in range(shape[0])]


# Rows 0 and 3 of ``runs`` hold several entries, so their level-0 coordinate
# repeats in COO storage; row 1 is empty.
_RANK2_CASES = {
    "runs": (
        (4, 4),
        [
            [1.0, 0.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
            [5.0, 6.0, 0.0, 7.0],
        ],
    ),
    "empty": ((2, 3), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    "single": ((3, 2), [[0.0, 0.0], [0.0, 9.0], [0.0, 0.0]]),
    "full": ((2, 2), [[1.0, 2.0], [3.0, 4.0]]),
    "zero-extent": ((0, 3), []),
}

# ``runs`` stores three entries under level-0 coordinate 0 and four under 2,
# and repeats a level-1 coordinate inside both; ``dense[1]`` is all zero.
_RANK3_CASES = {
    "runs": (
        (3, 2, 3),
        [
            [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[4.0, 0.0, 0.0], [5.0, 6.0, 7.0]],
        ],
    ),
    "empty": ((2, 2, 2), [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]]),
    "single": ((2, 2, 2), [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 8.0], [0.0, 0.0]]]),
}

_RANK1_CASES = {
    "runs": ((5,), [0.0, 2.0, 0.0, 3.0, 4.0]),
    "empty": ((3,), [0.0, 0.0, 0.0]),
    "single": ((3,), [0.0, 0.0, 6.0]),
}

_STORED_CASES = [
    pytest.param(shape, dense, id=f"rank{len(shape)}-{name}")
    for cases in (_RANK1_CASES, _RANK2_CASES, _RANK3_CASES)
    for name, (shape, dense) in cases.items()
]


def _identity(shape):
    return tuple(range(len(shape)))


@pytest.mark.parametrize("shape,dense", _STORED_CASES)
def test_coordinate_copy_into_dense_matches_the_compressed_program(shape, dense):
    modes = _identity(shape)
    compressed, _ = _run_stored_tensor_program(_COMPRESSED, dense, shape, modes, "copy")
    coordinate, _ = _run_stored_tensor_program(_COORDINATE, dense, shape, modes, "copy")
    assert coordinate == compressed == dense


@pytest.mark.parametrize("shape,dense", _STORED_CASES)
def test_coordinate_sum_into_dense_matches_the_compressed_program(shape, dense):
    modes = _identity(shape)
    compressed, _ = _run_stored_tensor_program(_COMPRESSED, dense, shape, modes, "sum")
    coordinate, _ = _run_stored_tensor_program(_COORDINATE, dense, shape, modes, "sum")
    assert coordinate == compressed == _sum_over_mode_zero(dense, shape, modes)


def _as_coordinate_storage(tensor):
    """The canonical COO storage an assembled all-coordinate result holds."""

    assert type(tensor) is LevelTensor
    rank = len(tensor.shape)
    assert tensor.level_kinds == (_COORDINATE,) * rank
    assert tensor.positions == (None,) * rank
    return LevelTensorStorage(
        shape=tensor.shape,
        modes=tuple(range(rank)),
        levels=tuple(CoordinateLevel(coords) for coords in tensor.coordinates),
        values=tensor.values,
    )


@pytest.mark.parametrize("shape,dense", _STORED_CASES)
def test_coordinate_copy_into_a_coordinate_result_returns_the_input(shape, dense):
    modes = _identity(shape)
    from_coordinate, storage = _run_stored_tensor_program(
        _COORDINATE, dense, shape, modes, "append"
    )
    assert _as_coordinate_storage(from_coordinate) == storage
    assert _as_coordinate_storage(from_coordinate).to_dense() == dense
    # The same entries stored in compressed levels assemble the same result.
    from_compressed, _ = _run_stored_tensor_program(
        _COMPRESSED, dense, shape, modes, "append"
    )
    assert from_compressed == from_coordinate


@pytest.mark.parametrize("shape,dense", _STORED_CASES)
def test_coordinate_input_assembles_the_same_compressed_result(shape, dense):
    # The result kind does not change what a coordinate input contributes:
    # appended into compressed levels, both inputs give one level tensor.
    modes = _identity(shape)
    from_compressed, _ = _run_stored_tensor_program(
        _COMPRESSED, dense, shape, modes, "append", result_kind=_COMPRESSED
    )
    from_coordinate, _ = _run_stored_tensor_program(
        _COORDINATE, dense, shape, modes, "append", result_kind=_COMPRESSED
    )
    assert from_coordinate == from_compressed
    assert from_coordinate.values == tuple(
        value for _, value in _nonzero_cells(dense, shape)
    )


@pytest.mark.parametrize("shape,dense", _STORED_CASES)
def test_reduction_into_a_coordinate_result_merges_by_coordinate(shape, dense):
    modes = _identity(shape)
    sums = _sum_over_mode_zero(dense, shape, modes)
    stored = sorted({logical[0] for logical, _ in _nonzero_cells(dense, shape)})
    for kind in (_COMPRESSED, _COORDINATE):
        coordinate, _ = _run_stored_tensor_program(
            kind, dense, shape, modes, "reduce", result_kind=_COORDINATE
        )
        assert coordinate.level_kinds == (_COORDINATE,)
        assert coordinate.positions == (None,)
        assert coordinate.coordinates == (tuple(stored),)
        assert coordinate.values == tuple(sums[row] for row in stored)
        compressed, _ = _run_stored_tensor_program(
            kind, dense, shape, modes, "reduce", result_kind=_COMPRESSED
        )
        assert compressed.coordinates == coordinate.coordinates
        assert compressed.values == coordinate.values


def _mode_orders():
    for shape in ((5,), (4, 5), (3, 4, 2), (1, 1), (3, 1, 4), (2, 0, 3)):
        for modes in itertools.permutations(range(len(shape))):
            yield pytest.param(shape, modes, id=f"{shape}-{modes}".replace(" ", ""))


@pytest.mark.parametrize("shape,modes", list(_mode_orders()))
def test_coordinate_programs_match_compressed_on_random_grids(shape, modes):
    rng = random.Random(20261012 + 97 * len(shape) + 7 * sum(shape) + modes[0])
    for density in (0.0, 0.15, 0.5, 1.0):
        for _ in range(3):
            dense = _random_dense(rng, shape, density)
            for result, expected in (
                ("copy", dense),
                ("sum", _sum_over_mode_zero(dense, shape, modes)),
            ):
                compressed, _ = _run_stored_tensor_program(
                    _COMPRESSED, dense, shape, modes, result
                )
                coordinate, _ = _run_stored_tensor_program(
                    _COORDINATE, dense, shape, modes, result
                )
                assert coordinate == compressed == expected


def _zeros_of_shape(shape):
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [_zeros_of_shape(shape[1:]) for _ in range(shape[0])]


def _permuted_dense(dense, shape, modes):
    """The tensor whose logical mode ``modes[l]`` is mode ``l`` of ``dense``."""

    rank = len(shape)
    permuted_shape = [0] * rank
    for level, mode in enumerate(modes):
        permuted_shape[mode] = shape[level]
    result = _zeros_of_shape(tuple(permuted_shape))
    for logical, value in _nonzero_cells(dense, shape):
        target = result
        moved = [0] * rank
        for level, mode in enumerate(modes):
            moved[mode] = logical[level]
        for coordinate in moved[:-1]:
            target = target[coordinate]
        target[moved[-1]] = value
    return tuple(permuted_shape), result


@pytest.mark.parametrize("kind", (_COMPRESSED, _COORDINATE), ids=lambda k: k.value)
@pytest.mark.parametrize(
    "shape,dense",
    [
        pytest.param(*_RANK2_CASES["runs"], id="rank2"),
        pytest.param(*_RANK3_CASES["runs"], id="rank3"),
    ],
)
def test_permuting_the_stored_modes_permutes_the_result(kind, shape, dense):
    # One set of level arrays, read under every assignment of logical modes
    # to levels: the copy is the stored tensor with its modes permuted the
    # same way.  For rank 2 the one non-identity assignment is the transpose.
    rank = len(shape)
    stored = LevelTensorStorage.from_dense(
        dense, shape, _identity(shape), (kind,) * rank
    )
    for modes in itertools.permutations(range(rank)):
        permuted_shape, expected = _permuted_dense(dense, shape, modes)
        permuted = LevelTensorStorage(
            shape=permuted_shape,
            modes=modes,
            levels=stored.levels,
            values=stored.values,
        )
        program, a, out = _stored_tensor_program(kind, modes, "copy")
        result = _execute_unverified(program, {a: permuted}, {out: permuted_shape})
        assert result[out] == expected
        if modes == (1, 0):
            assert permuted_shape == (shape[1], shape[0])
            assert expected == [list(column) for column in zip(*dense)]
        if modes == _identity(shape):
            assert expected == dense


def _merge_program(kind, mode):
    """``C[i, j] = A[i, j] op B[i, j]`` by merging both levels of A and B.

    The level-0 merge binds each operand's position, and each level-1 cursor
    takes its own operand's bound position as its parent.  UNION adds with
    zero defaults; INTERSECTION multiplies.
    """

    builder = _Builder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    dimensions = (dim_i.dimension, dim_j.dimension)
    a, b, c = (builder.new_symbol_id() for _ in range(3))

    def stored_levels():
        return (builder.level(kind, 0), builder.level(kind, 1))

    decl_a = builder.tensor(a, "A", ScalarType.FLOAT32, dimensions, stored_levels())
    decl_b = builder.tensor(b, "B", ScalarType.FLOAT32, dimensions, stored_levels())
    decl_c = builder.tensor(
        c, "C", ScalarType.FLOAT32, dimensions, builder.dense_levels(2)
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    rows_a, rows_b, cols_a, cols_b = (builder.new_cursor_id() for _ in range(4))
    position_a = builder.new_position_id()
    position_b = builder.new_position_id()
    union = mode is MergeMode.UNION

    def default():
        return builder.float_const(0.0) if union else None

    leaf = builder.store(
        c,
        (builder.index_value(row), builder.index_value(col)),
        builder.binary(
            BinaryOp.ADD if union else BinaryOp.MUL,
            builder.cursor_value(cols_a, default()),
            builder.cursor_value(cols_b, default()),
        ),
    )
    inner = builder.merged_sparse_for(
        mode,
        (
            builder.sparse_cursor(cols_a, a, 1, builder.position_value(position_a)),
            builder.sparse_cursor(cols_b, b, 1, builder.position_value(position_b)),
        ),
        col,
        builder.block((leaf,)),
    )
    outer = builder.merged_sparse_for(
        mode,
        (
            builder.sparse_cursor(rows_a, a, 0, builder.root_position()),
            builder.sparse_cursor(rows_b, b, 0, builder.root_position()),
        ),
        row,
        builder.block((inner,)),
        (position_a, position_b),
    )
    program = builder.program(
        (dim_i, dim_j),
        (decl_a, decl_b, decl_c),
        (a, b),
        (c,),
        builder.block((outer,)),
    )
    return program, a, b, c


@pytest.mark.parametrize("mode", (MergeMode.UNION, MergeMode.INTERSECTION))
def test_coordinate_merges_match_compressed_merges(mode):
    rng = random.Random(20261012 + len(mode.value))
    shape = (5, 6)
    grids = [
        (_RANK2_CASES["runs"][1], _RANK2_CASES["runs"][1]),
        (_zeros_of_shape((4, 4)), _RANK2_CASES["runs"][1]),
        (_RANK2_CASES["runs"][1], _zeros_of_shape((4, 4))),
    ]
    grids += [
        (_random_dense(rng, shape, left), _random_dense(rng, shape, right))
        for left in (0.1, 0.4, 0.9)
        for right in (0.1, 0.4, 0.9)
        for _ in range(3)
    ]
    for left, right in grids:
        grid_shape = (len(left), len(left[0]))
        results = {}
        for kind in (_COMPRESSED, _COORDINATE):
            program, a, b, c = _merge_program(kind, mode)
            inputs = {
                a: LevelTensorStorage.from_dense(left, grid_shape, (0, 1), (kind,) * 2),
                b: LevelTensorStorage.from_dense(
                    right, grid_shape, (0, 1), (kind,) * 2
                ),
            }
            results[kind] = _execute_unverified(program, inputs, {c: grid_shape})[c]
            if kind is _COMPRESSED:
                assert run_program(program, inputs, {c: grid_shape})[c] == results[kind]
        if mode is MergeMode.UNION:
            expected = [
                [x + y for x, y in zip(left_row, right_row)]
                for left_row, right_row in zip(left, right)
            ]
        else:
            expected = [
                [x * y for x, y in zip(left_row, right_row)]
                for left_row, right_row in zip(left, right)
            ]
        assert results[_COORDINATE] == results[_COMPRESSED] == expected


def _windowed_sum_program(kind, width):
    """``y[i] += A[i, j]`` with level 0 of A walked in panels of ``width``."""

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
        (builder.level(kind, 0), builder.level(kind, 1)),
    )
    decl_y = builder.tensor(
        y, "y", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    rows = builder.new_cursor_id()
    cols = builder.new_cursor_id()
    row_position = builder.new_position_id()
    col_position = builder.new_position_id()
    tile = builder.new_tile_id()
    inner = builder.sparse_for(
        builder.sparse_cursor(cols, a, 1, builder.position_value(row_position)),
        col_position,
        col,
        builder.block(
            (
                builder.store_reduce(
                    y,
                    (builder.index_value(row),),
                    ReduceOp.ADD,
                    builder.cursor_value(cols),
                ),
            )
        ),
    )
    window = builder.sparse_window_for(
        tile,
        builder.sparse_cursor(rows, a, 0, builder.root_position()),
        row_position,
        row,
        builder.block((inner,)),
    )
    panel = builder.panel_outer_for(
        tile, row, dim_i.dimension, width, y, 0, builder.block((window,))
    )
    program = builder.program(
        (dim_i, dim_j), (decl_a, decl_y), (a,), (y,), builder.block((panel,))
    )
    return program, a, y


@pytest.mark.parametrize("width", (1, 2, 3, 7))
def test_coordinate_window_visits_each_stored_coordinate_once(width):
    shape, dense = _RANK2_CASES["runs"]
    results = {}
    for kind in (_COMPRESSED, _COORDINATE):
        program, a, y = _windowed_sum_program(kind, width)
        storage = LevelTensorStorage.from_dense(dense, shape, (0, 1), (kind,) * 2)
        results[kind] = _execute_unverified(program, {a: storage}, {y: shape[:1]})[y]
    assert results[_COORDINATE] == results[_COMPRESSED] == [6.0, 0.0, 4.0, 18.0]


def test_position_load_reads_a_coordinate_leaf_position():
    # The position a last-level coordinate cursor binds is the entry index,
    # which is the position the value stream is addressed by.
    shape, dense = _RANK2_CASES["runs"]
    results = {}
    for kind in (_COMPRESSED, _COORDINATE):
        builder = _Builder()
        dim_i = builder.dimension("i")
        dim_j = builder.dimension("j")
        a = builder.new_symbol_id()
        c = builder.new_symbol_id()
        dimensions = (dim_i.dimension, dim_j.dimension)
        decl_a = builder.tensor(
            a,
            "A",
            ScalarType.FLOAT32,
            dimensions,
            (builder.level(kind, 0), builder.level(kind, 1)),
        )
        decl_c = builder.tensor(
            c, "C", ScalarType.FLOAT32, dimensions, builder.dense_levels(2)
        )
        row = builder.new_index_id()
        col = builder.new_index_id()
        rows = builder.new_cursor_id()
        cols = builder.new_cursor_id()
        row_position = builder.new_position_id()
        col_position = builder.new_position_id()
        inner = builder.sparse_for(
            builder.sparse_cursor(cols, a, 1, builder.position_value(row_position)),
            col_position,
            col,
            builder.block(
                (
                    builder.store(
                        c,
                        (builder.index_value(row), builder.index_value(col)),
                        builder.position_load(a, builder.position_value(col_position)),
                    ),
                )
            ),
        )
        outer = builder.sparse_for(
            builder.sparse_cursor(rows, a, 0, builder.root_position()),
            row_position,
            row,
            builder.block((inner,)),
        )
        program = builder.program(
            (dim_i, dim_j), (decl_a, decl_c), (a,), (c,), builder.block((outer,))
        )
        storage = LevelTensorStorage.from_dense(dense, shape, (0, 1), (kind,) * 2)
        results[kind] = _execute_unverified(program, {a: storage}, {c: shape})[c]
    assert results[_COORDINATE] == results[_COMPRESSED] == dense


def test_run_program_binds_a_coordinate_input(monkeypatch):
    # ``run_program`` verifies first, and the verifier's admission of
    # coordinate levels is tested in its own file.  Verification is replaced
    # by a no-op here so that the rest of the entry point runs on a
    # coordinate input: the input snapshot, the layout check, execution.
    import scorch.compiler.loopir.oracle as oracle_module

    monkeypatch.setattr(oracle_module, "verify_program", lambda program: None)
    shape, dense = _RANK2_CASES["runs"]
    program, a, out = _stored_tensor_program(_COORDINATE, (0, 1), "copy")
    storage = LevelTensorStorage.from_dense(dense, shape, (0, 1), (_COORDINATE,) * 2)
    assert run_program(program, {a: storage}, {out: shape})[out] == dense

    # The bound storage is snapshotted before the output shapes are read, so
    # state forged into the caller's instance during that read is not
    # observed.  The same forged instance is rejected when it is bound.
    program, a, out = _stored_tensor_program(_COORDINATE, (0, 1), "append")

    class ForgingOutputShapes(dict):
        def __getitem__(self, key):
            object.__setattr__(storage.levels[1], "coords", (0,) * len(storage.values))
            return super().__getitem__(key)

    result = run_program(program, {a: storage}, ForgingOutputShapes({out: shape}))
    assert storage.levels[1].coords == (0,) * len(storage.values)
    assert _as_coordinate_storage(result[out]).to_dense() == dense
    with pytest.raises(LoopIROracleError, match="input A has invalid level storage"):
        run_program(program, {a: storage}, {out: shape})


def test_coordinate_input_binding_boundaries_fail_closed():
    shape, dense = _RANK2_CASES["runs"]
    program, a, out = _stored_tensor_program(_COORDINATE, (0, 1), "copy")

    def run(bound):
        return _execute_unverified(program, {a: bound}, {out: shape})

    with pytest.raises(
        LoopIROracleError, match="must be bound to a LevelTensorStorage"
    ):
        run(dense)
    with pytest.raises(LoopIROracleError, match="not a canonical CSR declaration"):
        run(CsrMatrix.from_dense(dense))
    for kinds, modes in (
        ((_COMPRESSED, _COMPRESSED), (0, 1)),
        ((LevelKind.DENSE, _COMPRESSED), (0, 1)),
        ((_COORDINATE, _COORDINATE), (1, 0)),
    ):
        wrong_layout = LevelTensorStorage.from_dense(dense, shape, modes, kinds)
        with pytest.raises(LoopIROracleError, match="does not match its declaration"):
            run(wrong_layout)
    # A compressed declaration refuses coordinate storage the same way.
    compressed_program, a2, out2 = _stored_tensor_program(_COMPRESSED, (0, 1), "copy")
    coordinate = LevelTensorStorage.from_dense(dense, shape, (0, 1), (_COORDINATE,) * 2)
    with pytest.raises(LoopIROracleError, match="does not match its declaration"):
        run_program(compressed_program, {a2: coordinate}, {out2: shape})
    # The extents of a coordinate input bind the program's dimensions.
    with pytest.raises(LoopIROracleError, match="dimension extent mismatch"):
        _execute_unverified(program, {a: coordinate}, {out: (shape[0], shape[1] + 1)})


def test_coordinate_cursor_under_a_parent_that_starts_no_run_fails_closed():
    # A level-1 cursor of a COO tensor takes as its parent the entry that
    # starts a run on level 0.  Here a second level-1 cursor takes the
    # position the first level-1 loop binds instead.
    shape, dense = _RANK2_CASES["runs"]
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
        (builder.level(_COORDINATE, 0), builder.level(_COORDINATE, 1)),
    )
    decl_y = builder.tensor(
        y, "y", ScalarType.FLOAT32, (dim_i.dimension,), builder.dense_levels(1)
    )
    row = builder.new_index_id()
    col = builder.new_index_id()
    inner_col = builder.new_index_id()
    rows, cols, again = (builder.new_cursor_id() for _ in range(3))
    row_position, col_position, again_position = (
        builder.new_position_id() for _ in range(3)
    )
    # Entry 0 starts the run of row 0, so the first selection succeeds.
    # Entry 1 shares row 0 with it and starts no run, so the second fails.
    misuse = builder.sparse_for(
        builder.sparse_cursor(again, a, 1, builder.position_value(col_position)),
        again_position,
        inner_col,
        builder.block(
            (
                builder.store_reduce(
                    y,
                    (builder.index_value(row),),
                    ReduceOp.ADD,
                    builder.cursor_value(again),
                ),
            )
        ),
    )
    inner = builder.sparse_for(
        builder.sparse_cursor(cols, a, 1, builder.position_value(row_position)),
        col_position,
        col,
        builder.block((misuse,)),
    )
    outer = builder.sparse_for(
        builder.sparse_cursor(rows, a, 0, builder.root_position()),
        row_position,
        row,
        builder.block((inner,)),
    )
    program = builder.program(
        (dim_i, dim_j), (decl_a, decl_y), (a,), (y,), builder.block((outer,))
    )
    storage = LevelTensorStorage.from_dense(dense, shape, (0, 1), (_COORDINATE,) * 2)
    with pytest.raises(
        LoopIROracleError,
        match="cursor segment selection failed on A: parent position 1 does not",
    ):
        _execute_unverified(program, {a: storage}, {y: shape[:1]})


def test_coordinate_result_boundaries_fail_closed():
    shape, dense = _RANK2_CASES["runs"]
    # Column-major storage walks the entries in an order that is not the
    # logical order a coordinate result is appended in.
    program, a, out = _stored_tensor_program(_COORDINATE, (1, 0), "append")
    by_columns = LevelTensorStorage.from_dense(dense, shape, (1, 0), (_COORDINATE,) * 2)
    with pytest.raises(LoopIROracleError, match="lexicographically increasing"):
        _execute_unverified(program, {a: by_columns}, {out: shape})

    def result_program(levels):
        builder = _Builder()
        dim_i = builder.dimension("i")
        dim_j = builder.dimension("j")
        x = builder.new_symbol_id()
        c = builder.new_symbol_id()
        dimensions = (dim_i.dimension, dim_j.dimension)
        decl_x = builder.tensor(
            x, "x", ScalarType.FLOAT32, dimensions, builder.dense_levels(2)
        )
        decl_c = builder.tensor(
            c,
            "C",
            ScalarType.FLOAT32,
            dimensions,
            tuple(builder.level(kind, mode) for kind, mode in levels),
        )
        row = builder.new_index_id()
        col = builder.new_index_id()
        coords = (builder.index_value(row), builder.index_value(col))
        append = builder.append_entry(c, coords, builder.load(x, coords))
        program = builder.program(
            (dim_i, dim_j),
            (decl_x, decl_c),
            (x,),
            (c,),
            builder.block(
                (
                    builder.dense_for(
                        row,
                        dim_i.dimension,
                        builder.block(
                            (
                                builder.dense_for(
                                    col, dim_j.dimension, builder.block((append,))
                                ),
                            )
                        ),
                    ),
                )
            ),
        )
        return program, x, c

    # A dense loop nest appends every cell, zeros included, in logical order.
    program, x, c = result_program(((_COORDINATE, 0), (_COORDINATE, 1)))
    result = _execute_unverified(program, {x: [[1.0, 0.0], [0.0, 2.0]]}, {c: (2, 2)})[c]
    assert result.coordinates == ((0, 0, 1, 1), (0, 1, 0, 1))
    assert result.values == (1.0, 0.0, 0.0, 2.0)

    # The assembled container has no mode permutation.
    program, x, c = result_program(((_COORDINATE, 1), (_COORDINATE, 0)))
    with pytest.raises(LoopIROracleError, match="permuted mode order"):
        _execute_unverified(program, {x: [[1.0]]}, {c: (1, 1)})

    # A coordinate level beside any other kind is not a result layout.
    for levels in (
        ((_COORDINATE, 0), (_COMPRESSED, 1)),
        ((LevelKind.DENSE, 0), (_COORDINATE, 1)),
    ):
        program, x, c = result_program(levels)
        with pytest.raises(LoopIROracleError, match="output C cannot be assembled"):
            _execute_unverified(program, {x: [[1.0]]}, {c: (1, 1)})
