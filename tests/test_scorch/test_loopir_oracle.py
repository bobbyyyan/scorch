"""Differential and fail-closed coverage for the production LoopIR oracle.

The oracle executes verified dense programs over plain Python containers and
is compared exactly against independent pure-Python references — both sides
use Python-float arithmetic in the same order, so no tolerances are involved.
"""

import random

import pytest

from scorch.compiler.identity import SymbolId
from scorch.compiler.loopir.nodes import BinaryOp, ScalarType
from scorch.compiler.loopir.oracle import LoopIROracleError, run_program

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
    # A zero inner extent is inferable from nested values ((2, 0)); a zero
    # outer extent ((0, n)) is not, which is the same recorded nested-value
    # inference boundary the Phase-3.5 spike documented.
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


def test_uninferable_zero_leading_extent_fails_closed():
    fixture = build_matmul()
    with pytest.raises(LoopIROracleError) as error:
        run_program(
            fixture.program,
            {fixture.a: [], fixture.b: []},
            {fixture.c: (0, 0)},
        )
    assert "cannot be inferred" in str(error.value)


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
