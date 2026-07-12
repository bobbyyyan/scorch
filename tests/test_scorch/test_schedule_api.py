from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from scorch.compiler.cin import (
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
)
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.scheduler import (
    RelayoutSpec,
    Schedule,
    Scheduler,
    TileSpec,
    get_forced_schedule,
    regblock_force,
    schedule_force,
)
from scorch.ops import _codegen_kernel_cache_key, _einsum_cache_key


def _build_spmm(loop_order=("i", "j", "k")) -> ForAll:
    index_vars = {name: IndexVar(name) for name in ("i", "j", "k")}
    i, j, k = (index_vars[name] for name in ("i", "j", "k"))

    c = TensorVar("C", fmt="dd")
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="dd")
    c[i, k] = a[i, j] * b[j, k]

    stmt = c._assignment
    for name in reversed(loop_order):
        stmt = ForAll(index_vars[name], stmt)

    assert isinstance(stmt, ForAll)
    return stmt


def _build_sddmm() -> ForAll:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    out = TensorVar("D", fmt="oo")
    mask = TensorVar("A", fmt="ds")
    left = TensorVar("B", fmt="dd")
    right = TensorVar("C", fmt="dd")
    assignment = TensorAssign(
        out[i, j],
        mask[i, j] * left[i, k] * right[j, k],
        op=Operation.ADD,
    )
    return ForAll(i, ForAll(j, ForAll(k, assignment)))


def _build_named_spmm(
    *,
    result_name="Output",
    sparse_name="SparseInput",
    dense_name="DenseInput",
    row_name="row",
    panel_name="reduce",
    pack_name="free",
    dense_format="dd",
    dtype=torch.float32,
) -> ForAll:
    row = IndexVar(row_name)
    panel = IndexVar(panel_name)
    pack = IndexVar(pack_name)
    result = TensorVar(result_name, fmt="dd", dtype=dtype)
    sparse = TensorVar(sparse_name, fmt="ds", dtype=dtype)
    dense = TensorVar(dense_name, fmt=dense_format, dtype=dtype)
    result[row, pack] = sparse[row, panel] * dense[panel, pack]
    return ForAll(row, ForAll(panel, ForAll(pack, result._assignment)))


def _packed_schedule(
    *,
    row="i",
    panel="j",
    pack="k",
    operand="B",
    nc=4,
    jc=3,
    scope=None,
    accum="direct",
) -> Schedule:
    if scope is None:
        scope = panel
    return Schedule(
        loop_order=(row, panel, pack),
        tiles=(
            TileSpec(
                pack,
                nc,
                placement="outermost",
                accum=accum,
                unroll=False,
            ),
            TileSpec(
                panel,
                jc,
                placement=f"child_of:{pack}_out",
                kind="panel",
                accum="direct",
            ),
        ),
        relayout=RelayoutSpec(operand, pack, nc, scope_var=scope),
        tag=f"packed-tile-ijk-{scope}-{accum}",
        parallel_loop=row,
    )


def _lower_to_cpp(stmt: ForAll) -> str:
    lowered = CINLowerer().lower_IndexStmt(stmt)
    return LLIRLowerer().lower_llir(lowered)


def _loop_chain(stmt):
    names = []
    body = stmt
    while isinstance(body, ForAll):
        names.append(body.index_var.name)
        body = body.stmt
    return names, body


def test_empty_schedule_is_identical_to_auto_schedule():
    with regblock_force(False):
        auto_scheduled = Scheduler.auto_schedule(_build_spmm())
        explicitly_scheduled = Scheduler.apply_schedule(_build_spmm(), Schedule())

        assert str(explicitly_scheduled) == str(auto_scheduled)
        assert _lower_to_cpp(explicitly_scheduled) == _lower_to_cpp(auto_scheduled)


def test_apply_schedule_honors_explicit_loop_order_exactly():
    schedule = Schedule(loop_order=("i", "k", "j"), tag="ikj")

    with regblock_force(False):
        scheduled = Scheduler.apply_schedule(_build_spmm(), schedule)

    loop_names, body = _loop_chain(scheduled)
    assert loop_names == ["i", "k", "j"]
    assert isinstance(body, TensorAssign)


@pytest.mark.parametrize(
    "loop_order",
    [
        ("i", "j", "unknown"),
        ("i", "j"),
        ("i", "j", "j"),
    ],
    ids=("unknown", "missing", "duplicate"),
)
def test_apply_schedule_rejects_invalid_loop_orders(loop_order):
    with pytest.raises(ValueError):
        Scheduler.apply_schedule(
            _build_spmm(),
            Schedule(loop_order=loop_order),
        )


@pytest.mark.parametrize(
    "loop_order",
    [("j", "k", "i"), ("k", "i", "j"), ("k", "j", "i")],
)
def test_apply_schedule_rejects_result_child_before_parent(loop_order):
    with pytest.raises(ValueError, match="result storage order"):
        Scheduler.apply_schedule(
            _build_spmm(),
            Schedule(loop_order=loop_order),
        )


def test_tile_k_width_and_child_placement_reach_generated_cpp():
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(TileSpec("k", 4, placement="child_of:i"),),
        tag="tile-k-4-child-of-i",
    )

    with regblock_force(False):
        scheduled = Scheduler.apply_schedule(_build_spmm(), schedule)
        cpp = _lower_to_cpp(scheduled)

    loop_names, body = _loop_chain(scheduled)
    assert loop_names == ["i", "k_out"]
    assert isinstance(body, Where)

    producer_loop_names, producer_body = _loop_chain(body.producer)
    assert producer_loop_names == ["j", "k_in"]
    assert isinstance(producer_body, TensorAssign)

    tile_size_vars = scheduled.get_tile_size_vars()
    assert len(tile_size_vars) == 1
    assert tile_size_vars[0].index_var.name == "k"
    assert tile_size_vars[0].size == 4

    assert "constexpr int kTile_k = 4;" in cpp
    assert "k_out += kTile_k" in cpp
    assert "int64_t k = k_out + k_in;" in cpp
    assert "if (k >= B1_size)" in cpp
    assert "if (k >= C1_size)" in cpp
    assert cpp.index("for (int64_t i = 0") < cpp.index("k_out = 0")


def test_schedule_cache_key_is_canonical_and_complete():
    base = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(TileSpec("k", 4, placement="child_of:i", unroll=True),),
        tag="tile-k",
        parallel_loop="i",
    )
    equivalent = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(TileSpec("k", 4, placement="child_of:i", unroll=True),),
        tag="tile-k",
        parallel_loop="i",
    )

    assert base.cache_key == equivalent.cache_key
    base_tile = base.tiles[0]
    variants = [
        replace(base, loop_order=("i", "k", "j")),
        replace(base, tiles=(replace(base_tile, index_var="i"),)),
        replace(base, tiles=(replace(base_tile, width=8),)),
        replace(base, tiles=(replace(base_tile, placement="outermost"),)),
        replace(
            base,
            tiles=(replace(base_tile, parallel=True),),
            parallel_loop=None,
        ),
        replace(base, tiles=(replace(base_tile, kind="panel"),)),
        replace(base, tiles=(replace(base_tile, accum="heap"),)),
        replace(base, tiles=(replace(base_tile, unroll=False),)),
        replace(
            base,
            relayout=RelayoutSpec(operand="B", pack_var="k", strip_width=32),
        ),
        replace(base, tag="different-tag"),
        replace(base, parallel_loop="k_out"),
    ]

    assert all(base.cache_key != variant.cache_key for variant in variants)
    assert len({variant.cache_key for variant in variants}) == len(variants)


def test_schedule_discriminates_both_codegen_caches():
    width_four = Schedule(tiles=(TileSpec("k", 4),), tag="same-human-tag")
    width_eight = Schedule(tiles=(TileSpec("k", 8),), tag="same-human-tag")
    cin = _build_spmm()

    class FakeTensor:
        format = "d,s"
        dtype = "float32"

    tensors = (FakeTensor(), FakeTensor())
    assert _einsum_cache_key("ij,jk->ik", tensors, "dd", None, width_four) != (
        _einsum_cache_key("ij,jk->ik", tensors, "dd", None, width_eight)
    )
    assert _codegen_kernel_cache_key(cin, None, width_four) != (
        _codegen_kernel_cache_key(cin, None, width_eight)
    )


def test_codegen_cache_key_discriminates_generated_tensor_dtypes():
    float_cin = _build_spmm()
    double_cin = _build_spmm()
    for access in double_cin.tensor_accesses:
        access.tensor.dtype = torch.float64
    schedule = _packed_schedule()

    assert str(float_cin) == str(double_cin)
    assert _codegen_kernel_cache_key(float_cin, None, schedule) != (
        _codegen_kernel_cache_key(double_cin, None, schedule)
    )


def test_schedule_force_is_context_local_and_restores_nested_values():
    outer = Schedule(tag="outer")
    inner = Schedule(tag="inner")
    assert get_forced_schedule() is None
    with schedule_force(outer):
        assert get_forced_schedule() == outer
        with schedule_force(inner):
            assert get_forced_schedule() == inner
        assert get_forced_schedule() == outer
    assert get_forced_schedule() is None


def test_tile_i_generates_a_real_affine_stripmine():
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(TileSpec("i", 4, placement="outermost", accum="direct"),),
        tag="tile-i-4",
    )

    with regblock_force(False):
        scheduled = Scheduler.apply_schedule(_build_spmm(), schedule)
        cpp = _lower_to_cpp(scheduled)

    loop_names, _ = _loop_chain(scheduled)
    assert loop_names[:2] == ["i_out", "i_in"]

    assert "constexpr int kTile_i = 4;" in cpp
    assert "i_out += kTile_i" in cpp
    assert "i_in < kTile_i" in cpp
    assert "int64_t i = i_out + i_in;" in cpp
    assert "if (i >= A0_size)" in cpp
    assert "int pA0 = i;" in cpp
    assert "int pC0 = i;" in cpp
    assert cpp.count("int pA0 = i;") == 1
    assert cpp.count("int pC0 = i;") == 1
    assert "for (int64_t i = 0; i < A0_size; i++)" not in cpp

    resolve_pos = cpp.index("int64_t i = i_out + i_in;")
    tail_guard_pos = cpp.index("if (i >= A0_size)", resolve_pos)
    assert resolve_pos < tail_guard_pos < cpp.index("int pA0 = i;")
    assert resolve_pos < tail_guard_pos < cpp.index("int pC0 = i;")


def test_panel_tile_lowers_to_windowed_sparse_iteration():
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(TileSpec("j", 32, kind="panel", accum="direct"),),
        tag="tile-j-panel",
        parallel_loop="i",
    )

    scheduled = Scheduler.apply_schedule(_build_spmm(), schedule)
    cpp = _lower_to_cpp(scheduled)

    assert "constexpr int kTile_j = 32;" in cpp
    assert "j_out += kTile_j" in cpp
    assert "j_out_end = std::min(j_out + kTile_j, B0_size)" in cpp
    assert cpp.count("std::lower_bound") == 2
    assert "pA1 = pA1_panel_begin" in cpp
    assert cpp.index("j_out = 0") < cpp.index("#pragma omp parallel for")
    assert cpp.index("#pragma omp parallel for") < cpp.index("i = 0")


def test_panel_outermost_placement_hoists_over_affine_tile():
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec("k", 4, accum="direct", unroll=False),
            TileSpec("j", 8, kind="panel", accum="direct"),
        ),
        tag="panel-outside-k",
        parallel_loop="i",
    )

    scheduled = Scheduler.apply_schedule(_build_spmm(), schedule)
    cpp = _lower_to_cpp(scheduled)

    assert cpp.index("j_out = 0") < cpp.index("k_out = 0")
    assert cpp.index("k_out = 0") < cpp.index("#pragma omp parallel for")


def test_parallel_free_tile_uses_safe_work_estimate():
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                4,
                parallel=True,
                accum="direct",
                unroll=False,
            ),
        ),
        tag="parallel-k-tile",
    )

    cpp = _lower_to_cpp(Scheduler.apply_schedule(_build_spmm(), schedule))

    tile_count = "((B1_size + kTile_k - 1) / kTile_k)"
    assert f"scorch_nthreads(-1, {tile_count})" in cpp
    assert f"scorch_chunk({tile_count}, -1)" in cpp
    assert "A1_pos[B1_size]" not in cpp


def test_unsupported_or_unsafe_schedule_requests_are_rejected():
    with pytest.raises(NotImplementedError, match="Affine reduction tiling"):
        Scheduler.apply_schedule(
            _build_sddmm(),
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(
                    TileSpec(
                        "k",
                        4,
                        placement="child_of:j",
                        accum="direct",
                    ),
                ),
            ),
        )

    with pytest.raises(NotImplementedError, match="accum='direct'"):
        Scheduler.apply_schedule(
            _build_spmm(),
            Schedule(tiles=(TileSpec("j", 4, kind="panel"),)),
        )

    with pytest.raises(NotImplementedError, match="Stack accumulation"):
        Scheduler.apply_schedule(
            _build_spmm(),
            Schedule(tiles=(TileSpec("i", 4),)),
        )

    with pytest.raises(NotImplementedError, match="compressed tensor access"):
        Scheduler.apply_schedule(
            _build_spmm(),
            Schedule(tiles=(TileSpec("k", 4, kind="panel", accum="direct"),)),
        )

    with pytest.raises(ValueError, match="Reduction loops"):
        Scheduler.apply_schedule(
            _build_spmm(),
            Schedule(parallel_loop="j"),
        )

    with pytest.raises(ValueError, match="ragged-tail break"):
        Scheduler.apply_schedule(
            _build_spmm(),
            Schedule(
                tiles=(TileSpec("k", 4, accum="direct"),),
                parallel_loop="k_in",
            ),
        )

    with pytest.raises(ValueError, match="CSR dense-parent row loop"):
        Scheduler.apply_schedule(
            _build_spmm(),
            Schedule(
                tiles=(TileSpec("j", 4, kind="panel", accum="direct"),),
            ),
        )

    with pytest.raises(ValueError, match="row loop to precede"):
        Scheduler.apply_schedule(
            _build_spmm(),
            Schedule(
                loop_order=("j", "i", "k"),
                tiles=(TileSpec("j", 4, kind="panel", accum="direct"),),
                parallel_loop="i",
            ),
        )

    with pytest.raises(NotImplementedError, match="at_depth"):
        Scheduler.apply_schedule(
            _build_spmm(),
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(
                    TileSpec(
                        "j",
                        4,
                        placement="at_depth:0",
                        kind="panel",
                        accum="direct",
                    ),
                ),
                parallel_loop="i",
            ),
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: TileSpec("k", 4, parallel="yes"),
        lambda: TileSpec("k", 4, unroll=1),
        lambda: TileSpec("k", 4, placement="child_of:"),
        lambda: TileSpec("k", 4, placement="at_depth:-1"),
        lambda: RelayoutSpec("B", "k", True),
        lambda: RelayoutSpec("B", "k", 4, scope_var=1),
        lambda: RelayoutSpec("B", "k", 4, scope_var=""),
        lambda: Schedule(parallel_loop=1),
    ],
)
def test_schedule_constructor_validation(factory):
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_storage_schedule_fields_are_immutable_and_legacy_scope_is_canonical():
    relayout = RelayoutSpec("B", "k", 4, scope_var="k")
    with pytest.raises(FrozenInstanceError):
        relayout.scope_var = "j"

    legacy = replace(
        _packed_schedule(),
        relayout=RelayoutSpec("B", "k", 4),
    )
    explicit_panel = _packed_schedule(scope="j")
    assert legacy.relayout is not None
    assert legacy.relayout.scope_var == "j"
    assert legacy.cache_key == explicit_panel.cache_key


def test_packed_relayout_is_structural_and_name_independent():
    schedule = _packed_schedule(
        row="row",
        panel="reduce",
        pack="free",
        operand="DenseInput",
        nc=4,
        jc=3,
    )

    scheduled = Scheduler.apply_schedule(_build_named_spmm(), schedule)
    cpp = _lower_to_cpp(scheduled)

    allocation = "std::vector<float> packed_DenseInput_storage"
    pack_loop = (
        "for (int64_t reduce_pack = reduce_out; "
        "reduce_pack < reduce_out_end; reduce_pack++)"
    )
    row_loop = "for (int64_t row = 0; row < SparseInput0_size; row++)"
    packed_read = "packed_DenseInput[(reduce - reduce_out) * " "kTile_free + free_in]"

    assert allocation in cpp
    assert "packed_DenseInput_storage.data()" in cpp
    assert pack_loop in cpp
    assert packed_read in cpp
    assert "DenseInput_val[pDenseInput1]" not in cpp
    assert cpp.index("free_out = 0") < cpp.index("reduce_out = 0")
    assert cpp.index(pack_loop) < cpp.index(row_loop)
    assert cpp.count("scorch_zero_dense(Output_values, Output_capacity)") == 1
    assert "Output_values[pOutput1] +=" in cpp


def test_full_stage_and_heap_result_are_structural_and_name_independent():
    schedule = _packed_schedule(
        row="row",
        panel="reduce",
        pack="free",
        operand="DenseInput",
        scope="free",
        accum="heap",
    )

    cpp = _lower_to_cpp(Scheduler.apply_schedule(_build_named_spmm(), schedule))

    full_stage = (
        "std::vector<float> packed_DenseInput_storage("
        "(size_t) DenseInput0_size * (size_t) kTile_free);"
    )
    compact_result = (
        "std::vector<float> tiled_Output_storage("
        "(size_t) (Output0_size) * (size_t) kTile_free);"
    )
    pack_loop = (
        "for (int64_t reduce_pack = 0; "
        "reduce_pack < DenseInput0_size; reduce_pack++)"
    )
    panel_loop = (
        "for (int64_t reduce_out = 0; reduce_out < DenseInput0_size; "
        "reduce_out += kTile_reduce)"
    )
    assert full_stage in cpp
    assert compact_result in cpp
    assert cpp.index(pack_loop) < cpp.index(panel_loop)
    assert "packed_DenseInput[reduce * kTile_free + free_in]" in cpp
    assert "tiled_Output[pOutput0 * kTile_free + free_in] +=" in cpp
    assert "Output_values[pOutput1] +=" not in cpp
    assert "scorch_zero_dense(Output_values, Output_capacity)" not in cpp


def test_packed_relayout_generates_hygienic_staging_names():
    pointer_collision_stmt = _build_named_spmm(pack_name="packed_DenseInput")
    pointer_collision_schedule = _packed_schedule(
        row="row",
        panel="reduce",
        pack="packed_DenseInput",
        operand="DenseInput",
    )
    pointer_cpp = _lower_to_cpp(
        Scheduler.apply_schedule(
            pointer_collision_stmt,
            pointer_collision_schedule,
        )
    )

    assert "float* __restrict__ packed_DenseInput_1 =" in pointer_cpp
    assert "packed_DenseInput_1[(reduce - reduce_out)" in pointer_cpp
    assert "float* __restrict__ packed_DenseInput =" not in pointer_cpp

    loop_collision_stmt = _build_named_spmm(
        panel_name="k_pack",
        pack_name="k",
    )
    loop_collision_schedule = _packed_schedule(
        row="row",
        panel="k_pack",
        pack="k",
        operand="DenseInput",
    )
    loop_cpp = _lower_to_cpp(
        Scheduler.apply_schedule(loop_collision_stmt, loop_collision_schedule)
    )

    assert "for (int64_t k_pack_1 = 0; k_pack_1 < kTile_k; k_pack_1++)" in loop_cpp
    assert "(k_pack - k_pack_out) * kTile_k + k_in" in loop_cpp


def test_packed_and_unpacked_schedules_cannot_alias_either_cache():
    packed = _packed_schedule()
    unpacked = replace(packed, relayout=None, tag=packed.tag)
    cin = _build_spmm()

    class FakeTensor:
        format = "d,s"
        dtype = "float32"

    tensors = (FakeTensor(), FakeTensor())
    assert _einsum_cache_key("ij,jk->ik", tensors, "dd", None, packed) != (
        _einsum_cache_key("ij,jk->ik", tensors, "dd", None, unpacked)
    )
    assert _codegen_kernel_cache_key(cin, None, packed) != (
        _codegen_kernel_cache_key(cin, None, unpacked)
    )


def test_all_storage_strategy_crosses_have_distinct_schedule_and_codegen_keys():
    schedules = [
        replace(_packed_schedule(scope=scope, accum=accum), tag="same-human-tag")
        for scope in ("j", "k")
        for accum in ("direct", "heap")
    ]
    cin = _build_spmm()

    class FakeTensor:
        format = "d,s"
        dtype = "float32"

    tensors = (FakeTensor(), FakeTensor())
    assert len({schedule.cache_key for schedule in schedules}) == 4
    assert (
        len(
            {
                _einsum_cache_key("ij,jk->ik", tensors, "dd", None, schedule)
                for schedule in schedules
            }
        )
        == 4
    )
    assert (
        len({_codegen_kernel_cache_key(cin, None, schedule) for schedule in schedules})
        == 4
    )


@pytest.mark.parametrize(
    ("schedule", "error", "message"),
    [
        (
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(TileSpec("j", 3, kind="panel", accum="direct"),),
                relayout=RelayoutSpec("B", "k", 4),
                parallel_loop="i",
            ),
            ValueError,
            "affine tile for pack_var",
        ),
        (
            replace(
                _packed_schedule(),
                relayout=RelayoutSpec("B", "k", 8),
            ),
            ValueError,
            "strip_width must match",
        ),
        (
            replace(
                _packed_schedule(),
                relayout=RelayoutSpec("A", "k", 4),
            ),
            ValueError,
            "pack_var.*does not index",
        ),
        (
            replace(
                _packed_schedule(),
                tiles=(
                    TileSpec(
                        "k",
                        4,
                        placement="child_of:i",
                        accum="direct",
                        unroll=False,
                    ),
                    TileSpec(
                        "j",
                        3,
                        placement="child_of:k_out",
                        kind="panel",
                        accum="direct",
                    ),
                ),
            ),
            ValueError,
            "outermost affine tile",
        ),
        (
            _packed_schedule(scope="i"),
            ValueError,
            "scope_var must name either",
        ),
        (
            _packed_schedule(accum="stack"),
            NotImplementedError,
            "direct or heap-backed",
        ),
        (
            replace(
                _packed_schedule(accum="heap"),
                tiles=(
                    TileSpec(
                        "k",
                        4,
                        placement="child_of:i",
                        accum="heap",
                        unroll=False,
                    ),
                    TileSpec(
                        "j",
                        3,
                        placement="child_of:k_out",
                        kind="panel",
                        accum="direct",
                    ),
                ),
            ),
            ValueError,
            "Heap accumulation requires.*outermost",
        ),
        (
            Schedule(
                loop_order=("i", "j", "k"),
                tiles=(
                    TileSpec(
                        "k",
                        4,
                        placement="outermost",
                        parallel=True,
                        accum="heap",
                        unroll=False,
                    ),
                ),
            ),
            ValueError,
            "shared reusable storage",
        ),
    ],
    ids=(
        "missing-pack-tile",
        "mismatched-width",
        "wrong-operand",
        "placement",
        "invalid-scope",
        "stack-result",
        "heap-placement",
        "parallel-heap-tile",
    ),
)
def test_packed_relayout_rejects_incompatible_schedules(schedule, error, message):
    with pytest.raises(error, match=message):
        Scheduler.apply_schedule(_build_spmm(), schedule)


def test_heap_result_tile_rejects_outermost_panel_wrapper_before_lowering():
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                "k",
                4,
                placement="outermost",
                accum="heap",
                unroll=False,
            ),
            TileSpec(
                "j",
                3,
                placement="outermost",
                kind="panel",
                accum="direct",
            ),
        ),
        parallel_loop="i",
    )

    with pytest.raises(ValueError, match="requires.*remain outermost"):
        Scheduler.apply_schedule(_build_spmm(), schedule)


def test_packed_relayout_rejects_unsupported_format_and_dtype():
    named_schedule = _packed_schedule(
        row="row",
        panel="reduce",
        pack="free",
        operand="DenseInput",
    )
    with pytest.raises(NotImplementedError, match="rank-2 dense input"):
        Scheduler.apply_schedule(
            _build_named_spmm(dense_format="ds"),
            named_schedule,
        )
    with pytest.raises(NotImplementedError, match="float32 or float64"):
        Scheduler.apply_schedule(
            _build_named_spmm(dtype=torch.int32),
            named_schedule,
        )


def test_packed_relayout_rejects_ambiguous_repeated_staged_access():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    out = TensorVar("C", fmt="dd")
    sparse = TensorVar("A", fmt="ds")
    dense = TensorVar("B", fmt="dd")
    first_read = dense[j, k]
    second_read = dense[j, k]
    assignment = TensorAssign(
        out[i, k],
        sparse[i, j] * (first_read + second_read),
        op=Operation.ADD,
    )
    stmt = ForAll(i, ForAll(j, ForAll(k, assignment)))

    with pytest.raises(ValueError, match="exactly one input tensor access"):
        Scheduler.apply_schedule(stmt, _packed_schedule())


def test_packed_relayout_rejects_union_expression_before_lowering():
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    out = TensorVar("C", fmt="dd")
    sparse = TensorVar("A", fmt="ds")
    dense = TensorVar("B", fmt="dd")
    assignment = TensorAssign(
        out[i, k],
        sparse[i, j] + dense[j, k],
        op=Operation.ADD,
    )
    stmt = ForAll(i, ForAll(j, ForAll(k, assignment)))
    original = str(stmt)

    with pytest.raises(NotImplementedError, match="multiplicative contraction"):
        Scheduler.apply_schedule(stmt, _packed_schedule())
    assert str(stmt) == original


def test_packed_relayout_rejects_non_additive_assignment():
    stmt = _build_named_spmm()
    assignment = stmt
    while isinstance(assignment, ForAll):
        assignment = assignment.stmt
    assert isinstance(assignment, TensorAssign)
    assignment.op = Operation.SUB

    with pytest.raises(NotImplementedError, match="additive contraction"):
        Scheduler.apply_schedule(
            stmt,
            _packed_schedule(
                row="row",
                panel="reduce",
                pack="free",
                operand="DenseInput",
            ),
        )
