"""Execution and differential coverage for the Phase-3.5 LoopIR spike.

The three hand-authored feasibility programs run through the generic schema
and interpreter over plain-Python containers, and every result is compared
against an independent pure-Python dense reference.  The accumulation order
matches the reference exactly (adding a zero term never changes an IEEE
partial sum), so all comparisons are exact.
"""

import random

import pytest

from scorch.compiler.loopir_spike import interp as loopir_interp
from scorch.compiler.identity import new_index_id, new_symbol_id
from scorch.compiler.loopir_spike.csr import CsrFormatError, CsrMatrix
from scorch.compiler.loopir_spike.interp import LoopIRInterpreterError, run_program
from scorch.compiler.loopir_spike.nodes import (
    AppendEntry,
    Block,
    DenseFor,
    DimSize,
    FloatConst,
    IndexValue,
    IntConst,
    LevelKind,
    LoopProgram,
    SparseCursorDecl,
    SparseFor,
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

nid = new_loop_node_id


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


def run_spmv(dense_rows, n_cols, x):
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense(dense_rows, n_cols)
    results = run_program(
        fixture.program,
        {fixture.matrix: matrix, fixture.vector: x},
        {fixture.result: (matrix.n_rows,)},
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
    with pytest.raises(LoopIRInterpreterError, match="extent equality 1 failed"):
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


def test_shorter_second_operand_fails_closed():
    fixture = build_csr_union_add_program()
    a = CsrMatrix.from_dense([[1.0], [2.0], [3.0]], 1)
    b = CsrMatrix.from_dense([[1.0], [2.0]], 1)
    with pytest.raises(LoopIRInterpreterError, match="extent equality 0 failed"):
        run_program(
            fixture.program,
            {fixture.lhs: a, fixture.rhs: b},
            {fixture.result: (3, 1)},
        )


@pytest.mark.parametrize("vector", ([4.0], [4.0, 0.0, 0.0, 0.0]))
def test_spmv_vector_extent_mismatch_is_sparsity_independent(vector):
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[2.0, 0.0, 0.0]], 3)
    with pytest.raises(LoopIRInterpreterError, match="extent equality 1 failed"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: vector},
            {fixture.result: (1,)},
        )


def test_spmv_output_extent_mismatch_rejected_before_execution():
    fixture = build_csr_spmv_program()
    matrix = CsrMatrix.from_dense([[0.0, 0.0, 0.0]], 3)
    with pytest.raises(LoopIRInterpreterError, match="extent equality 0 failed"):
        run_program(
            fixture.program,
            {fixture.matrix: matrix, fixture.vector: [0.0, 0.0, 0.0]},
            {fixture.result: (2,)},
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
    with pytest.raises(LoopIRInterpreterError, match="extent equality 0 failed"):
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
    with pytest.raises(LoopIRInterpreterError, match="extent equality 0 failed"):
        run_program(
            fixture.program,
            {fixture.lhs: lhs, fixture.rhs: rhs},
            {fixture.result: (1, 1)},
        )


def test_elementwise_column_mismatch_rejected_with_empty_support():
    fixture = build_csr_intersection_multiply_program()
    lhs = CsrMatrix.from_dense([[0.0, 0.0, 0.0]], 3)
    rhs = CsrMatrix.from_dense([[0.0, 0.0]], 2)
    with pytest.raises(LoopIRInterpreterError, match="extent equality 1 failed"):
        run_program(
            fixture.program,
            {fixture.lhs: lhs, fixture.rhs: rhs},
            {fixture.result: (1, 3)},
        )


def test_elementwise_output_extent_mismatch_rejected_with_empty_support():
    fixture = build_csr_union_add_program()
    lhs = CsrMatrix.from_dense([], 0)
    rhs = CsrMatrix.from_dense([], 0)
    with pytest.raises(LoopIRInterpreterError, match="extent equality 1 failed"):
        run_program(
            fixture.program,
            {fixture.lhs: lhs, fixture.rhs: rhs},
            {fixture.result: (0, 1)},
        )


def _append_only_program(coords_list):
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
        (
            TensorDecl(
                nid(),
                result,
                "C",
                (LevelKind.DENSE, LevelKind.COMPRESSED),
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
    x = new_symbol_id()
    y = new_symbol_id()
    program = LoopProgram(
        nid(),
        (
            TensorDecl(nid(), x, "x", (LevelKind.DENSE,)),
            TensorDecl(nid(), y, "y", (LevelKind.DENSE,)),
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


def test_negative_extent_fails_closed():
    x = new_symbol_id()
    y = new_symbol_id()
    i = new_index_id()
    program = LoopProgram(
        nid(),
        (
            TensorDecl(nid(), x, "x", (LevelKind.DENSE,)),
            TensorDecl(nid(), y, "y", (LevelKind.DENSE,)),
        ),
        (x,),
        (y,),
        Block(
            nid(),
            (
                DenseFor(
                    nid(),
                    i,
                    IntConst(nid(), -2),
                    Block(
                        nid(),
                        (
                            Store(
                                nid(),
                                y,
                                (IndexValue(nid(), i),),
                                FloatConst(nid(), 1.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(LoopIRInterpreterError, match="negative dense extent"):
        run_program(program, {x: [1.0]}, {y: (1,)})


def test_unsupported_rank_one_compressed_cursor_fails_closed():
    x = new_symbol_id()
    y = new_symbol_id()
    j = new_index_id()
    cursor = SparseCursorDecl(
        node_id=nid(),
        cursor=new_cursor_id(),
        tensor=x,
        level=0,
        outer_indices=(),
    )
    program = LoopProgram(
        nid(),
        (
            TensorDecl(nid(), x, "x", (LevelKind.COMPRESSED,)),
            TensorDecl(nid(), y, "y", (LevelKind.DENSE,)),
        ),
        (x,),
        (y,),
        Block(
            nid(),
            (
                SparseFor(nid(), cursor, j, Block(nid(), ())),
                Store(
                    nid(),
                    y,
                    (IntConst(nid(), 0),),
                    FloatConst(nid(), 1.0),
                ),
            ),
        ),
    )
    with pytest.raises(LoopIRInterpreterError, match="unsupported"):
        run_program(program, {x: [1.0]}, {y: (1,)})


def test_run_program_verifies_first():
    from scorch.compiler.loopir_spike.verifier import LoopIRVerificationError

    program, result = _append_only_program([(0, 0)])
    forged = LoopProgram(
        program.node_id,
        program.tensors,
        program.inputs,
        (),
        program.body,
    )
    with pytest.raises(LoopIRVerificationError):
        run_program(forged, {}, {})


def test_dim_size_of_output_is_usable():
    x = new_symbol_id()
    y = new_symbol_id()
    i = new_index_id()
    program = LoopProgram(
        nid(),
        (
            TensorDecl(nid(), x, "x", (LevelKind.DENSE,)),
            TensorDecl(nid(), y, "y", (LevelKind.DENSE,)),
        ),
        (x,),
        (y,),
        Block(
            nid(),
            (
                DenseFor(
                    nid(),
                    i,
                    DimSize(nid(), y, 0),
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
    results = run_program(program, {x: [0.0]}, {y: (3,)})
    assert results[y] == [2.0, 2.0, 2.0]
