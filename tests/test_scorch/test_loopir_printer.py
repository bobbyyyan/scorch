"""Deterministic printing and canonical serialization for production LoopIR.

The binding contract: two independently constructed equivalent programs —
including ones whose global ``SymbolId``/``IndexId`` allocation histories
differ — print identically and serialize identically, semantically different
programs serialize differently, and neither surface accepts an unverified
program.  There is no deserializer, so no round-trip is claimed.
"""

from contextlib import contextmanager
import json
import sys

import pytest

from scorch.compiler.identity import new_index_id, new_symbol_id
from scorch.compiler.loopir.build import LoopIRBuilder
from scorch.compiler.loopir.nodes import BinaryOp, ReduceOp, ScalarType
from scorch.compiler.loopir.printer import (
    CANONICAL_SCHEMA,
    canonical_program_dump,
    print_program,
)
from scorch.compiler.loopir.verifier import (
    LoopIRVerificationError,
    MAX_LOOPIR_TILE_WIDTH,
)

from tests.test_scorch.test_loopir_verifier import (
    build_matmul,
    build_vector_add,
    forge,
)


@contextmanager
def minimum_int_string_digits():
    """Temporarily select CPython's smallest decimal-conversion allowance."""

    previous = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(sys.int_info.str_digits_check_threshold)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(previous)


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


def test_declaration_registry_order_is_not_semantic():
    program = build_matvec()
    canonical = canonical_program_dump(program)
    printed = print_program(program)
    forge(
        program,
        dimensions=tuple(reversed(program.dimensions)),
        tensors=tuple(reversed(program.tensors)),
    )
    assert canonical_program_dump(program) == canonical
    assert print_program(program) == printed


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
    # v8: the Phase-6 parallel slice added the program-level abstract
    # parallel selection, after v7 added the compact result-tile kinds
    # (result_tile_region, tiled_reduce), v6 the staged-operand kinds, v5
    # the sparse coordinate-window kinds, v4 the workspace node kinds, and
    # v3 the affine-split kinds.
    payload = json.loads(canonical_program_dump(build_matvec()))
    assert payload["schema"] == CANONICAL_SCHEMA == "scorch.loopir.canonical.v8"
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


# -- Phase-5 sparse subset ----------------------------------------------------

from scorch.compiler.loopir.nodes import MergeMode  # noqa: E402

from tests.test_scorch.test_loopir_verifier import (  # noqa: E402
    build_csr_spmv,
    build_union_add,
)


def test_sparse_canonical_dump_is_stable_across_global_id_histories():
    first = canonical_program_dump(build_csr_spmv().program)
    # Burn global identity allocations so the second construction's
    # SymbolId/IndexId values differ.
    for _ in range(7):
        build_union_add()
    second = canonical_program_dump(build_csr_spmv().program)
    assert first == second


def test_sparse_canonical_dump_distinguishes_merge_modes_and_defaults():
    union = canonical_program_dump(build_union_add().program)
    intersection = canonical_program_dump(
        build_union_add(mode=MergeMode.INTERSECTION, with_defaults=False).program
    )
    assert union != intersection
    assert '"mode":"union"' in union
    assert '"mode":"intersection"' in intersection
    assert '"kind":"float_const"' in union
    assert '"default":null' in intersection


def test_sparse_registry_permutation_is_not_semantic():
    fixture = build_union_add()
    baseline = canonical_program_dump(fixture.program)
    rendered = print_program(fixture.program)
    import dataclasses as _dataclasses

    program = fixture.program
    permuted = type(program)(
        program.node_id,
        tuple(reversed(program.dimensions)),
        tuple(reversed(program.tensors)),
        program.inputs,
        program.outputs,
        program.body,
    )
    del _dataclasses
    assert canonical_program_dump(permuted) == baseline
    assert print_program(permuted) == rendered


def test_sparse_printer_renders_positions_cursors_and_appends():
    rendered = print_program(build_csr_spmv().program)
    assert "sparse_for" in rendered
    assert "parent dense_pos(" in rendered
    assert "parent root" in rendered
    assert "value(c0)" in rendered
    merged = print_program(build_union_add().program)
    assert "merged_union_for" in merged
    assert "append t" in merged
    assert "default 0.0" in merged


def test_sparse_dumps_are_target_neutral():
    target_fragments = (
        "torch",
        "omp_",
        "#pragma",
        "std::",
        "int64_t",
        "__restrict",
        "emplace_back",
        "pragma",
        "scorch_",
        "1_pos[",
        "1_crd[",
        "_val[",
    )
    for program in (build_csr_spmv().program, build_union_add().program):
        dump = canonical_program_dump(program)
        rendered = print_program(program)
        for fragment in target_fragments:
            assert fragment not in dump
            assert fragment not in rendered


# -- Phase-6 affine-split printing/serialization ------------------------------


def build_tiled_matvec(width=4, unroll=False):
    from tests.test_scorch.test_loopir_verifier import build_tiled_matvec

    return build_tiled_matvec(width=width, unroll=unroll).program


def test_tiled_printer_renders_the_split_structure():
    text = print_program(build_tiled_matvec(width=4, unroll=True))
    # Identity renumbering follows body traversal: the split dimension 'j'
    # is seen first (by the origin loop), so it canonicalizes as d0.
    assert text == (
        "loopir.program {\n"
        "  dimension d0 'j'\n"
        "  dimension d1 'i'\n"
        "  tensor t0 'A' float32 dims(d1, d0) levels(dense@0, dense@1)\n"
        "  tensor t1 'x' float32 dims(d0) levels(dense@0)\n"
        "  tensor t2 'y' float32 dims(d1) levels(dense@0)\n"
        "  inputs(t0, t1)\n"
        "  outputs(t2)\n"
        "  body {\n"
        "    tile_outer_for s0 x0 in d0 width 4 {\n"
        "      for x1 in d1 {\n"
        "        tile_inner_for s0 x0 in d0 width 4 unroll {\n"
        "          store_reduce(add) t2[x1] = "
        "mul(load t0[x1, x0], load t1[x0])\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def test_tiled_canonical_dump_is_stable_across_global_id_histories():
    first = build_tiled_matvec()
    for _ in range(64):
        new_symbol_id()
        new_index_id()
    second = build_tiled_matvec()
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert print_program(first) == print_program(second)


def test_tiled_canonical_dump_distinguishes_schedule_facts():
    base = canonical_program_dump(build_tiled_matvec(width=4))
    assert canonical_program_dump(build_tiled_matvec(width=8)) != base
    assert canonical_program_dump(build_tiled_matvec(unroll=True)) != base
    payload = json.loads(base)
    outer = payload["body"]["statements"][0]
    assert outer["kind"] == "tile_outer_for"
    assert outer["width"] == 4
    inner = outer["body"]["statements"][0]["body"]["statements"][0]
    assert inner["kind"] == "tile_inner_for"
    assert inner["tile"] == outer["tile"] == 0
    assert inner["unroll"] is False


def test_tiled_surfaces_reject_unverified_programs():
    program = build_tiled_matvec()
    inner = program.body.statements[0].body.statements[0].body.statements[0]
    forge(inner, width=9)
    with pytest.raises(LoopIRVerificationError):
        canonical_program_dump(program)
    with pytest.raises(LoopIRVerificationError):
        print_program(program)


# -- Phase-6 workspace regions ------------------------------------------------


def build_stack_matmul(width=4):
    from tests.test_scorch.test_loopir_verifier import build_stack_matmul

    return build_stack_matmul(width=width).program


def test_workspace_printer_renders_the_region_structure():
    text = print_program(build_stack_matmul(width=4))
    # Identity renumbering follows body traversal: the split dimension 'k'
    # is seen first (by the origin loop), so it canonicalizes as d0, and the
    # region renders its producer and consumer roles explicitly.
    assert text == (
        "loopir.program {\n"
        "  dimension d0 'k'\n"
        "  dimension d1 'i'\n"
        "  dimension d2 'j'\n"
        "  tensor t0 'A' float32 dims(d1, d2) levels(dense@0, dense@1)\n"
        "  tensor t1 'B' float32 dims(d2, d0) levels(dense@0, dense@1)\n"
        "  tensor t2 'C' float32 dims(d1, d0) levels(dense@0, dense@1)\n"
        "  inputs(t0, t1)\n"
        "  outputs(t2)\n"
        "  body {\n"
        "    tile_outer_for s0 x0 in d0 width 4 {\n"
        "      for x1 in d1 {\n"
        "        workspace_region w0 'wksp' float32 over s0 {\n"
        "          producer {\n"
        "            for x2 in d2 {\n"
        "              tile_inner_for s0 x0 in d0 width 4 {\n"
        "                workspace_reduce(add) w0[x0] = "
        "mul(load t0[x1, x2], load t1[x2, x0])\n"
        "              }\n"
        "            }\n"
        "          }\n"
        "          consumer {\n"
        "            tile_inner_for s0 x0 in d0 width 4 {\n"
        "              store_reduce(add) t2[x1, x0] = w0[x0]\n"
        "            }\n"
        "          }\n"
        "        }\n"
        "      }\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def test_workspace_canonical_dump_is_stable_across_global_id_histories():
    first = build_stack_matmul()
    for _ in range(64):
        new_symbol_id()
        new_index_id()
    second = build_stack_matmul()
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert print_program(first) == print_program(second)


def test_workspace_canonical_dump_renumbers_raw_schedule_id_histories():
    from scorch.compiler.loopir.nodes import TileId, WorkspaceId

    first = build_stack_matmul()
    second = build_stack_matmul()
    outer = second.body.statements[0]
    row = outer.body.statements[0]
    region = row.body.statements[0]
    producer_inner = region.producer.statements[0].body.statements[0]
    consumer_inner = region.consumer.statements[0]
    replacement_tile = TileId(97)
    replacement_workspace = WorkspaceId(113)
    forge(outer, tile=replacement_tile)
    forge(producer_inner, tile=replacement_tile)
    forge(consumer_inner, tile=replacement_tile)
    forge(
        region.workspace,
        tile=replacement_tile,
        workspace=replacement_workspace,
    )
    forge(
        producer_inner.body.statements[0],
        workspace=replacement_workspace,
    )
    forge(
        consumer_inner.body.statements[0].value,
        workspace=replacement_workspace,
    )
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert print_program(first) == print_program(second)


def test_workspace_canonical_dump_serializes_the_region_semantics():
    base = canonical_program_dump(build_stack_matmul(width=4))
    assert canonical_program_dump(build_stack_matmul(width=8)) != base
    payload = json.loads(base)
    region = payload["body"]["statements"][0]["body"]["statements"][0]["body"][
        "statements"
    ][0]
    assert region["kind"] == "workspace_region"
    assert region["workspace"] == {"workspace": 0, "dtype": "float32", "tile": 0}
    producer_point = region["producer"]["statements"][0]["body"]["statements"][0]
    assert producer_point["kind"] == "tile_inner_for"
    reduce_stmt = producer_point["body"]["statements"][0]
    assert reduce_stmt["kind"] == "workspace_reduce"
    assert reduce_stmt["op"] == "add"
    assert reduce_stmt["workspace"] == 0
    consumer_point = region["consumer"]["statements"][0]
    copy_out = consumer_point["body"]["statements"][0]
    assert copy_out["kind"] == "store_reduce"
    assert copy_out["value"]["kind"] == "workspace_read"
    assert copy_out["value"]["workspace"] == 0


def test_workspace_display_name_is_not_semantic():
    program = build_stack_matmul()
    renamed = build_stack_matmul()
    region = renamed.body.statements[0].body.statements[0].body.statements[0]
    forge(region.workspace, name="scratch")
    assert canonical_program_dump(program) == canonical_program_dump(renamed)
    assert print_program(program) != print_program(renamed)


# -- Phase-6 sparse coordinate panels -----------------------------------------


def test_panel_printer_renders_the_window_structure():
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    text = print_program(build_panel_spmm(width=3).program)
    assert "panel_outer_for s0 x0 in d0 width 3 bound t1@0 {" in text
    assert (
        "sparse_window_for s0 (p0, x0) in c0 over t0 level 1 "
        "parent dense_pos(t0, level 0, parent root, coord x1) {"
    ) in text
    # The window replaces the plain sparse loop spelling entirely.
    assert "sparse_for (" not in text


def test_panel_canonical_dump_is_stable_across_global_id_histories():
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    first = build_panel_spmm().program
    for _ in range(64):
        new_symbol_id()
        new_index_id()
    second = build_panel_spmm().program
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert print_program(first) == print_program(second)


def test_panel_canonical_dump_renumbers_raw_schedule_id_histories():
    from scorch.compiler.loopir.nodes import TileId

    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    first = build_panel_spmm()
    second = build_panel_spmm()
    replacement = TileId(151)
    forge(second.panel, tile=replacement)
    forge(second.window, tile=replacement)
    assert canonical_program_dump(first.program) == canonical_program_dump(
        second.program
    )
    assert print_program(first.program) == print_program(second.program)


def test_panel_canonical_dump_serializes_the_window_semantics():
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    base = canonical_program_dump(build_panel_spmm(width=4).program)
    assert canonical_program_dump(build_panel_spmm(width=8).program) != base
    payload = json.loads(base)
    panel = payload["body"]["statements"][0]
    assert panel["kind"] == "panel_outer_for"
    assert panel["tile"] == 0
    assert panel["width"] == 4
    assert panel["bound_tensor"] == 1
    assert panel["bound_level"] == 0
    window = panel["body"]["statements"][0]["body"]["statements"][0]
    assert window["kind"] == "sparse_window_for"
    assert window["tile"] == 0
    assert window["cursor"]["tensor"] == 0
    assert window["cursor"]["level"] == 1


def test_panel_max_semantic_width_is_total_at_the_minimum_digit_limit():
    from tests.test_scorch.test_loopir_verifier import build_panel_spmm

    assert MAX_LOOPIR_TILE_WIDTH.bit_length() == 2048
    assert len(str(MAX_LOOPIR_TILE_WIDTH)) < sys.int_info.str_digits_check_threshold
    with minimum_int_string_digits():
        text = print_program(build_panel_spmm(width=MAX_LOOPIR_TILE_WIDTH).program)
        payload = json.loads(
            canonical_program_dump(
                build_panel_spmm(width=MAX_LOOPIR_TILE_WIDTH).program
            )
        )
    assert f"width {MAX_LOOPIR_TILE_WIDTH} " in text
    assert payload["body"]["statements"][0]["width"] == MAX_LOOPIR_TILE_WIDTH


def test_relayout_printer_renders_the_region_structure():
    from scorch.compiler.loopir.nodes import RelayoutScope

    from tests.test_scorch.test_loopir_verifier import build_relayout_spmm

    text = print_program(build_relayout_spmm(RelayoutScope.PANEL).program)
    assert "relayout_stage r0 t1 panel s1 pack s0 scope panel {" in text
    assert "staged r0[x1, x0]" in text
    # The staged read replaces the direct load spelling entirely.
    assert "load t1[" not in text

    text = print_program(build_relayout_spmm(RelayoutScope.PACK_AXIS).program)
    assert "relayout_stage r0 t1 panel s1 pack s0 scope pack_axis {" in text


def test_relayout_canonical_dump_is_stable_across_global_id_histories():
    from tests.test_scorch.test_loopir_verifier import build_relayout_spmm

    first = build_relayout_spmm().program
    for _ in range(64):
        new_symbol_id()
        new_index_id()
    second = build_relayout_spmm().program
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert print_program(first) == print_program(second)


def test_relayout_canonical_dump_renumbers_raw_schedule_id_histories():
    from scorch.compiler.loopir.nodes import RelayoutId

    from tests.test_scorch.test_loopir_verifier import build_relayout_spmm

    first = build_relayout_spmm()
    second = build_relayout_spmm()
    replacement = RelayoutId(151)
    forge(second.decl, relayout=replacement)
    forge(second.staged, relayout=replacement)
    assert canonical_program_dump(first.program) == canonical_program_dump(
        second.program
    )
    assert print_program(first.program) == print_program(second.program)


def test_relayout_canonical_dump_serializes_the_region_semantics():
    from scorch.compiler.loopir.nodes import RelayoutScope

    from tests.test_scorch.test_loopir_verifier import build_relayout_spmm

    base = canonical_program_dump(build_relayout_spmm(RelayoutScope.PANEL).program)
    assert (
        canonical_program_dump(build_relayout_spmm(RelayoutScope.PACK_AXIS).program)
        != base
    )
    payload = json.loads(base)
    stage = payload["body"]["statements"][0]["body"]["statements"][0]["body"][
        "statements"
    ][0]
    assert stage["kind"] == "relayout_stage"
    assert stage["relayout"]["relayout"] == 0
    assert stage["relayout"]["operand"] == 1
    assert stage["relayout"]["scope"] == "panel"
    assert stage["relayout"]["panel"] == 1
    assert stage["relayout"]["pack"] == 0
    leaf = stage["body"]["statements"][0]["body"]["statements"][0]["body"][
        "statements"
    ][0]["body"]["statements"][0]
    read = leaf["value"]["rhs"]
    assert read["kind"] == "staged_read"
    assert read["relayout"] == 0
    assert [index["kind"] for index in read["indices"]] == ["index", "index"]


def test_result_tile_printer_renders_the_region_structure():
    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    text = print_program(build_heap_spmm().program)
    assert "result_tile_region h0 t2 pack s0 {" in text
    assert "tiled_reduce(add) h0[x1, x0]" in text
    # The tiled reduce replaces the direct result-store spelling entirely.
    assert "store_reduce" not in text


def test_result_tile_canonical_dump_is_stable_across_global_id_histories():
    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    first = build_heap_spmm().program
    for _ in range(64):
        new_symbol_id()
        new_index_id()
    second = build_heap_spmm().program
    assert canonical_program_dump(first) == canonical_program_dump(second)
    assert print_program(first) == print_program(second)


def test_result_tile_canonical_dump_renumbers_raw_schedule_id_histories():
    from scorch.compiler.loopir.nodes import ResultTileId

    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    first = build_heap_spmm()
    second = build_heap_spmm()
    replacement = ResultTileId(151)
    forge(second.decl, result_tile=replacement)
    forge(second.leaf, result_tile=replacement)
    assert canonical_program_dump(first.program) == canonical_program_dump(
        second.program
    )
    assert print_program(first.program) == print_program(second.program)


def test_result_tile_canonical_dump_serializes_the_region_semantics():
    from tests.test_scorch.test_loopir_verifier import build_heap_spmm

    base = canonical_program_dump(build_heap_spmm().program)
    assert canonical_program_dump(build_heap_spmm(strip=5).program) != base
    payload = json.loads(base)
    region = payload["body"]["statements"][0]["body"]["statements"][0]
    assert region["kind"] == "result_tile_region"
    assert region["result_tile"]["result_tile"] == 0
    assert region["result_tile"]["result"] == 2
    assert region["result_tile"]["pack"] == 0
    leaf = region["body"]["statements"][0]["body"]["statements"][0]["body"][
        "statements"
    ][0]["body"]["statements"][0]
    assert leaf["kind"] == "tiled_reduce"
    assert leaf["result_tile"] == 0
    assert leaf["op"] == "add"
    assert [index["kind"] for index in leaf["indices"]] == ["index", "index"]


def test_canonical_dump_owns_the_parallel_selection():
    """The selection is semantic program state with a deterministic dump."""

    from scorch.compiler.loopir.nodes import ParallelPart

    from tests.test_scorch.test_loopir_verifier import (
        attach_selection,
        build_csr_spmv,
        build_stack_matmul,
    )

    def spmv_with_selection(nnz_level):
        fixture = build_csr_spmv()
        dim_i = fixture.program.tensors[2].dimensions[0]
        attach_selection(
            fixture,
            fixture.row,
            rows=dim_i,
            nnz=(
                None
                if nnz_level is None
                else fixture.builder.sparse_work_source(fixture.a, nnz_level)
            ),
        )
        return fixture.program

    bare = build_csr_spmv().program
    selected = spmv_with_selection(1)
    payload = json.loads(canonical_program_dump(selected))
    assert json.loads(canonical_program_dump(bare))["parallel"] is None
    assert payload["parallel"] == {
        "index": 0,
        "part": "logical",
        "discipline": "result_partition",
        "work": {"rows": 0, "nnz": {"tensor": 0, "level": 1}},
        "intent": "explicit",
    }
    # Deterministic across fresh builders and distinct from the bare and
    # uniform-work forms.
    assert canonical_program_dump(selected) == canonical_program_dump(
        spmv_with_selection(1)
    )
    assert canonical_program_dump(selected) != canonical_program_dump(bare)
    assert canonical_program_dump(selected) != canonical_program_dump(
        spmv_with_selection(None)
    )
    outer = build_stack_matmul()
    dim_k = outer.program.tensors[2].dimensions[1]
    attach_selection(outer, outer.col, part=ParallelPart.OUTER, rows=dim_k)
    outer_payload = json.loads(canonical_program_dump(outer.program))
    assert outer_payload["parallel"]["part"] == "outer"


def test_printer_renders_the_parallel_selection():
    from tests.test_scorch.test_loopir_verifier import (
        attach_selection,
        build_csr_spmv,
    )

    fixture = build_csr_spmv()
    dim_i = fixture.program.tensors[2].dimensions[0]
    attach_selection(
        fixture,
        fixture.row,
        rows=dim_i,
        nnz=fixture.builder.sparse_work_source(fixture.a, 1),
    )
    rendered = print_program(fixture.program)
    assert (
        "  parallel x0 part=logical discipline=result_partition "
        "work(d0, nnz(t0@1)) intent=explicit\n"
    ) in rendered
    assert "parallel" not in print_program(build_csr_spmv().program)
