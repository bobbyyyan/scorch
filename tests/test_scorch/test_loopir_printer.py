"""Deterministic printing and canonical serialization for production LoopIR.

The binding contract: two independently constructed equivalent programs —
including ones whose global ``SymbolId``/``IndexId`` allocation histories
differ — print identically and serialize identically, semantically different
programs serialize differently, and neither surface accepts an unverified
program.  There is no deserializer, so no round-trip is claimed.
"""

import json

import pytest

from scorch.compiler.identity import new_index_id, new_symbol_id
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.nodes import BinaryOp, ReduceOp, ScalarType
from scorch.compiler.loopir.printer import (
    CANONICAL_SCHEMA,
    canonical_program_dump,
    print_program,
)
from scorch.compiler.loopir.verifier import LoopIRVerificationError

from tests.test_scorch.test_loopir_verifier import (
    build_matmul,
    build_vector_add,
    forge,
)


def build_matvec(op=BinaryOp.MUL, dtype=ScalarType.FLOAT32):
    """y[i] += A[i, j] `op` x[j]; fresh global identities per call."""

    builder = LoopIRBuilder()
    dim_i = builder.dimension("i")
    dim_j = builder.dimension("j")
    a, x, y = (builder.new_symbol_id() for _ in range(3))
    decl_a = builder.tensor(
        a, "A", dtype, (dim_i.dimension, dim_j.dimension), builder.dense_levels(2)
    )
    decl_x = builder.tensor(x, "x", dtype, (dim_j.dimension,), builder.dense_levels(1))
    decl_y = builder.tensor(y, "y", dtype, (dim_i.dimension,), builder.dense_levels(1))
    index_i = builder.new_index_id()
    index_j = builder.new_index_id()
    leaf = builder.store_reduce(
        y,
        (builder.index_value(index_i),),
        ReduceOp.ADD,
        builder.binary(
            op,
            builder.load(
                a, (builder.index_value(index_i), builder.index_value(index_j))
            ),
            builder.load(x, (builder.index_value(index_j),)),
        ),
    )
    inner = builder.dense_for(index_j, dim_j.dimension, builder.block((leaf,)))
    outer = builder.dense_for(index_i, dim_i.dimension, builder.block((inner,)))
    return builder.program(
        (dim_i, dim_j),
        (decl_a, decl_x, decl_y),
        (a, x),
        (y,),
        builder.block((outer,)),
    )


def test_canonical_dump_is_stable_across_global_id_histories():
    first = build_matvec()
    # Burn through global identity allocations so the second construction
    # uses a disjoint SymbolId/IndexId range.
    for _ in range(64):
        new_symbol_id()
        new_index_id()
    second = build_matvec()
    assert first is not second
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert print_program(first) == print_program(second)


def test_canonical_dump_is_deterministic_per_program():
    program = build_matvec()
    assert canonical_program_dump(program) == canonical_program_dump(program)
    assert print_program(program) == print_program(program)


def test_canonical_dump_distinguishes_semantics():
    multiply = canonical_program_dump(build_matvec(BinaryOp.MUL))
    add = canonical_program_dump(build_matvec(BinaryOp.ADD))
    assert multiply != add
    f32 = canonical_program_dump(build_matvec(dtype=ScalarType.FLOAT32))
    f64 = canonical_program_dump(build_matvec(dtype=ScalarType.FLOAT64))
    assert f32 != f64


def test_canonical_dump_omits_display_names():
    program = build_matvec()
    renamed = build_matvec()
    for decl in renamed.dimensions:
        forge(decl, name=f"axis_{decl.name}")
    for decl in renamed.tensors:
        forge(decl, name=f"tensor_{decl.name}")
    assert canonical_program_dump(program) == canonical_program_dump(renamed)
    assert print_program(program) != print_program(renamed)


def test_canonical_dump_carries_schema_version():
    payload = json.loads(canonical_program_dump(build_matvec()))
    assert payload["schema"] == CANONICAL_SCHEMA == "scorch.loopir.canonical.v1"
    assert payload["inputs"] == [0, 1]
    assert payload["outputs"] == [2]
    assert payload["body"]["kind"] == "block"


def test_printer_renders_the_complete_program():
    text = print_program(build_matvec())
    assert text == (
        "loopir.program {\n"
        "  dimension d0 'i'\n"
        "  dimension d1 'j'\n"
        "  tensor t0 'A' float32 dims(d0, d1) levels(dense@0, dense@1)\n"
        "  tensor t1 'x' float32 dims(d1) levels(dense@0)\n"
        "  tensor t2 'y' float32 dims(d0) levels(dense@0)\n"
        "  inputs(t0, t1)\n"
        "  outputs(t2)\n"
        "  body {\n"
        "    for x0 in d0 {\n"
        "      for x1 in d1 {\n"
        "        store_reduce(add) t2[x0] = "
        "mul(load t0[x0, x1], load t1[x1])\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def test_store_and_matmul_render_deterministically():
    assert "store t2[x0] = add(load t0[x0], load t1[x0])" in print_program(
        build_vector_add().program
    )
    assert "store_reduce(add) t2[x0, x2]" in print_program(build_matmul().program)


def test_both_surfaces_fail_closed_on_invalid_programs():
    fixture = build_vector_add()
    forge(fixture.program.tensors[1], dtype=ScalarType.FLOAT64)
    with pytest.raises(LoopIRVerificationError):
        canonical_program_dump(fixture.program)
    with pytest.raises(LoopIRVerificationError):
        print_program(fixture.program)
