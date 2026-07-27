"""Codegen unit tests for the Phase 2b register-block dual-path (codegen-parity).

The dual-path emits ONE format-keyed kernel that branches at runtime on the
free/output dim size:

    if (B1_size <= REGBLOCK_MAX_N) { <register-block, single stack tile> }
    else                          { <byte-identical baseline nest> }

so narrow-k gets the register-block win (output tile held in a stack-local
``wksp[]`` across the row's nonzeros) while wide-k keeps baseline parity *by
construction* (the register-block arm provably cannot fire for large N). The
branch is evaluated once, outside the parallel region.

These tests are gate-INDEPENDENT: they drive the register-block path explicitly
via ``regblock_force(...)`` rather than relying on the ``SCORCH_REGBLOCK`` env
default, so they behave identically whether or not the default is flipped. They
therefore never perturb the direct-``auto_schedule`` schedule-shape tests, which
assert the baseline nest.
"""

import copy
from dataclasses import replace

import pytest
import torch

import scorch.ops as ops
from scorch.ops import matmul
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.lower_llir import LoopIRTargetError
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir
from scorch.compiler.scheduler import (
    Schedule,
    Scheduler,
    regblock_force,
    _regblock_max_n,
    _regblock_tile_width,
)


def _build_spmm_cin(
    *,
    sparse_format="ds",
    bind_shapes=False,
    explicit_update=False,
):
    """Unscheduled CIN for CSR(A: ds) @ dense(B: dd) -> dense(C: dd).

    Index layout matches the real ``einsum("ij,jk->ik")`` SpMM: contraction over
    ``j``, free/output dim ``k`` -> the generated free-dim symbols are
    ``kTile_k`` / ``k_out`` / ``k_in`` and the runtime branch is on ``B1_size``.
    """
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dd")  # [i, k]
    a = TensorVar("A", fmt=sparse_format)
    b = TensorVar("B", fmt="dd")  # [j, k]
    if bind_shapes:
        c.shape, c.dtype = (4, 6), torch.float32
        a.shape, a.dtype = (4, 5), torch.float32
        b.shape, b.dtype = (5, 6), torch.float32
    if explicit_update:
        assignment = TensorAssign(
            c[i, k],
            a[i, j] * b[j, k],
            op=Operation.ADD,
        )
    else:
        c[i, k] = a[i, j] * b[j, k]
        assignment = c._assignment
    return ForAll(i, ForAll(j, ForAll(k, assignment)))


def _lower_to_cpp(fn_or_cin) -> str:
    return LLIRLowerer().lower_llir(fn_or_cin)


def test_dual_path_emits_runtime_branch_with_both_arms():
    """The dual-path kernel is a single runtime free-dim branch whose then-arm is
    the register-block nest (stack workspace) and whose else-arm is the baseline
    memory-destination nest (hoisted dense pointer, direct output ``+=``)."""
    built = ops._build_regblock_dual_path(_build_spmm_cin(), None)
    assert built is not None, "register-block dual-path should apply to CSR@dense SpMM"
    fn, key = built
    cpp = _lower_to_cpp(fn)

    cutoff = _regblock_max_n()
    # exactly one runtime branch on the free-dim size (B1_size), one per kernel
    assert cpp.count("if (B1_size <=") == 1
    assert f"if (B1_size <= {cutoff})" in cpp
    # both arms are a parallel row loop -> two omp regions, branch splits them
    assert cpp.count("#pragma omp parallel for") == 2

    # then-arm: register-block (stack workspace accumulated across the reduction)
    assert "wksp" in cpp
    assert "kTile_k" in cpp
    assert "wksp[k_in] += A_val" in cpp
    # else-arm: baseline memory-destination nest (pointer-hoisted, direct +=)
    assert "_B_val_ptr" in cpp
    assert "C_values[pC1] += A_val" in cpp

    # cache key is distinguished so it never collides with the plain kernel
    assert key.endswith("|rbdual")


def test_dual_path_single_tile_width_tracks_cutoff():
    """The register-block arm is a SINGLE tile: the tile width equals the free-dim
    tile-width knob (== REGBLOCK_MAX_N by default), so N <= cutoff never
    re-traverses the sparse row (no ragged-tail second pass)."""
    fn, _ = ops._build_regblock_dual_path(_build_spmm_cin(), None)
    cpp = _lower_to_cpp(fn)
    assert f"constexpr int kTile_k = {_regblock_tile_width()};" in cpp
    # ragged-tail guard present in BOTH the producer accumulate and consumer flush
    assert cpp.count("if (k >= B1_size)") >= 1  # producer bound
    assert cpp.count("if (k >= C1_size)") >= 1  # consumer flush bound


def test_regblock_off_is_plain_baseline_nest():
    """With the register-block path forced OFF the schedule is the plain baseline:
    no workspace, no tile, no runtime branch. This is the gate-OFF guarantee that
    keeps the direct-``auto_schedule`` schedule-shape tests green."""
    with regblock_force(False):
        scheduled = Scheduler.auto_schedule(_build_spmm_cin())
        cpp = _lower_to_cpp(CINLowerer().lower_IndexStmt(scheduled))
    assert "wksp" not in cpp
    assert "kTile_" not in cpp
    assert "if (B1_size <=" not in cpp
    # baseline still parallel + restrict-qualified
    assert "#pragma omp parallel for" in cpp
    assert "__restrict__" in cpp


@pytest.mark.parametrize("n", [1, 8, 16, 64])
def test_dual_path_correct_both_sides_of_cutoff(n):
    """The runtime branch is numerically correct for N on both sides of the cutoff:
    N <= 8 takes the register-block arm, N >= 16 the baseline else-arm. Both must
    match a dense reference (repo convention: atol=rtol=1e-3)."""
    torch.manual_seed(0)
    m = k = 128
    a = torch.randn(m, k, dtype=torch.float32)
    a = a * (torch.rand(m, k) < 0.1)  # ~90% sparse
    a_csr = a.to_sparse_csr()
    b = torch.randn(k, n, dtype=torch.float32)
    ref = a @ b

    # Force the dual-path regardless of the global default; clear the shape-agnostic
    # dispatch cache first so the forced (dual) module is built rather than a
    # previously-cached baseline module being reused (the dispatch key is
    # regblock-agnostic by design).
    ops._einsum_dispatch_cache.clear()
    try:
        with regblock_force(True):
            out = matmul(a_csr, b, use_cache=False)
    finally:
        ops._einsum_dispatch_cache.clear()

    assert torch.allclose(out.to(torch.float32), ref, atol=1e-3, rtol=1e-3)


_CENSUS_DIMS = {"i": 4, "j": 5, "k": 6, "l": 3}


def _census_operand(spec):
    """Normalize a census operand spec to (fmt, indices, mode_order|None)."""

    if len(spec) == 2:
        return spec[0], spec[1], None
    return spec


def _census_physical_shape(indices, mode_order):
    """Physical level extents: the logical extents viewed through the order."""

    logical = tuple(_CENSUS_DIMS[x] for x in indices)
    if mode_order is None:
        return logical
    return tuple(logical[mode] for mode in mode_order)


def _build_census_cin(fmt_result, result_indices, operands, nest, *, bind_shapes=False):
    """One qualifying-domain census CIN with optional bound physical shapes."""

    from scorch.compiler.cin import BinaryOp as CINBinaryOp

    ivars = {name: IndexVar(name) for name in "ijkl"}
    result = TensorVar("C", fmt=fmt_result)
    if bind_shapes:
        result.shape = tuple(_CENSUS_DIMS[x] for x in result_indices)
        result.dtype = torch.float32
    rhs = None
    for position, operand_spec in enumerate(operands):
        fmt, indices, mode_order = _census_operand(operand_spec)
        operand = TensorVar("ABDE"[position], fmt=fmt)
        if mode_order is not None:
            operand.mode_order = list(mode_order)
        if bind_shapes:
            operand.shape = _census_physical_shape(indices, mode_order)
            operand.dtype = torch.float32
        access = operand[tuple(ivars[x] for x in indices)]
        rhs = access if rhs is None else CINBinaryOp(Operation.MUL, rhs, access)
    assignment = TensorAssign(
        result[tuple(ivars[x] for x in result_indices)],
        rhs,
        op=Operation.ADD,
    )
    statement = assignment
    for name in reversed(nest):
        statement = ForAll(ivars[name], statement)
    return statement


_DENSE_DUAL_CONSTITUENTS = {
    "batched_dense": ("ddd", "lik", (("ddd", "lij"), ("ddd", "ljk")), "lijk"),
    "batched_transposed": (
        "ddd",
        "lik",
        (("ddd", "lij"), ("ddd", "lkj", (0, 2, 1))),
        "lijk",
    ),
    "dense_matmul": ("dd", "ik", (("dd", "ij"), ("dd", "jk")), "ijk"),
    "four_operand_ds": (
        "dd",
        "ik",
        (("ds", "ij"), ("dd", "jk"), ("dd", "ik"), ("d", "k")),
        "ijk",
    ),
    "transposed_spmm": (
        "dd",
        "ik",
        (("ds", "ij"), ("dd", "kj", (1, 0))),
        "ijk",
    ),
    "ttm_dense": ("ddd", "ijk", (("ddd", "ijl"), ("dd", "lk")), "ijlk"),
    "three_operand_ds": ("dd", "ik", (("ds", "ij"), ("dd", "jk"), ("dd", "ik")), "ijk"),
}


@pytest.mark.parametrize("family", sorted(_DENSE_DUAL_CONSTITUENTS))
def test_dense_dual_constituents_compose_from_actual_loopir_arms(family):
    """Representative identity-layout dual kernels reconstruct from LoopIR arms.

    The reduce-out migration closed the dense-matmul constituent of the
    production dual domain.  Cover rank-2, batched, rank-3, and three-/four-
    operand qualifying families: stitching the two actual LoopIR-produced
    arm lowerings must reproduce the production ``_build_regblock_dual_path``
    kernel byte-for-byte.  Comparing the same legacy helpers to themselves
    is not evidence; both arms here come from ``compile_cin_via_loopir``.
    """

    spec = _DENSE_DUAL_CONSTITUENTS[family]
    options = CompileOptions.from_environment(environ={})
    built = ops._build_regblock_dual_path(
        _build_census_cin(*spec, bind_shapes=True),
        None,
        compile_options=options,
    )
    assert built is not None
    production_cpp = _lower_to_cpp(built[0])

    arms = []
    for enabled in (False, True):
        arm_options = replace(
            options.with_regblock_enabled(enabled),
            requested_schedule=Schedule(),
        )
        bindings = tuple(
            (
                _census_physical_shape(
                    _census_operand(operand_spec)[1],
                    _census_operand(operand_spec)[2],
                ),
                torch.float32,
            )
            for operand_spec in spec[2]
        )
        arms.append(
            compile_cin_via_loopir(
                _build_census_cin(*spec),
                tuple(_CENSUS_DIMS[x] for x in spec[1]),
                bindings,
                compile_options=arm_options,
            )
        )
    base_arm, regblock_arm = arms
    assert base_arm.schedule.plan.auto_policy.regblock_enabled is False
    assert regblock_arm.schedule.plan.auto_policy.regblock_enabled is True
    reconstructed = ops._stitch_regblock_dual_path(
        copy.deepcopy(regblock_arm.llir_function),
        copy.deepcopy(base_arm.llir_function),
        options.scheduler.regblock_max_n,
    )
    assert reconstructed is not None
    assert _lower_to_cpp(reconstructed) == production_cpp


@pytest.mark.parametrize(
    ("family", "spec", "expected"),
    [
        (
            "spmm_ss",
            ("dd", "ik", (("ss", "ij"), ("dd", "jk")), "ijk"),
            "unsupported_program_shape",
        ),
        (
            "spmm_coo",
            ("dd", "ik", (("oo", "ij"), ("dd", "jk")), "ijk"),
            "unsupported_format",
        ),
        (
            "dense_at_csr",
            ("dd", "ik", (("dd", "ij"), ("ds", "jk")), "ijk"),
            "unsupported_schedule_auto_family",
        ),
        (
            "ttm_sparse_operand",
            ("ddd", "ijk", (("dds", "ijl"), ("dd", "lk")), "ijlk"),
            "unsupported_schedule_auto_family",
        ),
        (
            "batched_sparse_operand",
            ("ddd", "lik", (("dds", "lij"), ("ddd", "ljk")), "lijk"),
            "unsupported_schedule_auto_family",
        ),
        (
            "batched_transposed_sparse_operand",
            ("ddd", "lik", (("dds", "lij"), ("ddd", "lkj", (0, 2, 1))), "lijk"),
            "unsupported_schedule_auto_family",
        ),
    ],
)
def test_dual_domain_census_locks_open_boundaries(family, spec, expected):
    """Qualifying dual families outside the migrated arms fail closed.

    This representative production census locks the known format-family
    boundaries: hierarchical-compressed ``ss`` operands still fail target
    parent-position descent
    (``unsupported_program_shape``); COO operands fail level lowering
    (``unsupported_format``); trailing-compressed operands derive
    sparse-workspace-adjacent automatic plans that the family gate keeps on
    the legacy path (``unsupported_schedule_auto_family``).  Production
    release behavior is unchanged for every one of these: dispatch still
    builds the dual kernel from the legacy helpers, and the strangler path
    is not live.
    """

    options = CompileOptions.from_environment(environ={})
    assert (
        ops._build_regblock_dual_path(
            _build_census_cin(*spec),
            None,
            compile_options=options,
        )
        is not None
    )
    for enabled in (False, True):
        arm_options = replace(
            options.with_regblock_enabled(enabled),
            requested_schedule=Schedule(),
        )
        bindings = tuple(
            (
                _census_physical_shape(
                    _census_operand(operand_spec)[1],
                    _census_operand(operand_spec)[2],
                ),
                torch.float32,
            )
            for operand_spec in spec[2]
        )
        with pytest.raises(Exception) as error:
            compile_cin_via_loopir(
                _build_census_cin(*spec),
                tuple(_CENSUS_DIMS[x] for x in spec[1]),
                bindings,
                compile_options=arm_options,
            )
        assert getattr(error.value, "defect").code == expected


def test_transposed_dual_constituent_matches_production_alignment():
    """The census transposed entries are exactly the production alignment.

    Public einsum alignment derives each operand's desired mode order from
    the cost-selected loop order (``_bind_frontend_operand_mode_orders``):
    the order of the operand's subscripts within the selected loop order,
    spelled as logical-axis positions.  Deriving that decision from the
    production formula for the census programs proves the transposed
    families composed above are the real release-reachable layouts, not
    hand-picked permutations.
    """

    from scorch.compiler.scheduler import Scheduler

    # transposed_spmm follows the cost-selected branch; batched_transposed
    # follows the requested-schedule branch (an explicit public loop order),
    # exactly the two alignment branches production owns.
    cases = {
        "transposed_spmm": (("ij", "kj"), None),
        "batched_transposed": (("lij", "lkj"), ("l", "i", "j", "k")),
    }
    for family, (input_index_strs, requested_order) in cases.items():
        spec = _DENSE_DUAL_CONSTITUENTS[family]
        identity_operands = tuple(
            (fmt, indices)
            for fmt, indices, _ in (
                _census_operand(operand_spec) for operand_spec in spec[2]
            )
        )
        prealignment = _build_census_cin(
            spec[0], spec[1], identity_operands, spec[3], bind_shapes=True
        )
        if requested_order is None:
            selected = Scheduler.select_loop_order(
                prealignment,
                costs=CompileOptions.from_environment(environ={}).scheduler.cost_model,
            )
        else:
            selected = Scheduler.resolve_loop_order(prealignment, requested_order)
        selected_names = [index_var.name for index_var in selected]
        for operand_spec, subscripts in zip(spec[2], input_index_strs):
            _, _, census_order = _census_operand(operand_spec)
            derived = [
                subscripts.index(name) for name in selected_names if name in subscripts
            ]
            expected = (
                list(census_order)
                if census_order is not None
                else list(range(len(subscripts)))
            )
            assert derived == expected, (family, subscripts, derived)


def test_dual_domain_census_locks_non_qualifying_families():
    """Families where the regblock arm changes nothing build no dual."""

    options = CompileOptions.from_environment(environ={})
    non_qualifying = [
        ("d", "i", (("ds", "ij"), ("d", "j")), "ij"),
        ("d", "i", (("dd", "ij"), ("d", "j")), "ij"),
        ("dd", "ik", (("ds", "ij"), ("ds", "jk")), "ijk"),
        ("dd", "ij", (("dd", "ij"), ("dd", "ik"), ("dd", "jk")), "ijk"),
    ]
    for spec in non_qualifying:
        assert (
            ops._build_regblock_dual_path(
                _build_census_cin(*spec),
                None,
                compile_options=options,
            )
            is None
        )


@pytest.mark.parametrize(
    "explicit_update",
    [False, True],
    ids=["public_implicit_update", "explicit_add"],
)
def test_dual_path_composes_exactly_the_two_loopir_arm_lowerings(explicit_update):
    """The production dual stitch contains no third schedule beyond its arms.

    Both spellings of the public ds SpMM dual route — the frontend's
    implicit reduction (``TensorAssign.op is None``, normalized once at the
    CIN-to-LoopIR ownership boundary) and its explicit-ADD twin — must
    reconstruct the production dual kernel byte-for-byte from the two
    actual LoopIR-produced arm lowerings.  The runtime free-dim branch
    itself remains target-lowering state owned by Phase 7.
    """

    options = CompileOptions.from_environment(environ={})
    input_bindings = (
        ((4, 5), torch.float32),
        ((5, 6), torch.float32),
    )
    loopir_arms = []
    for enabled in (False, True):
        arm_options = replace(
            options.with_regblock_enabled(enabled),
            requested_schedule=Schedule(),
        )
        loopir_arms.append(
            compile_cin_via_loopir(
                _build_spmm_cin(explicit_update=explicit_update),
                (4, 6),
                input_bindings,
                compile_options=arm_options,
            )
        )

    built = ops._build_regblock_dual_path(
        _build_spmm_cin(bind_shapes=True, explicit_update=explicit_update),
        None,
        compile_options=options,
    )
    assert built is not None
    production_fn, _ = built
    production_cpp = _lower_to_cpp(production_fn)

    base_arm, regblock_arm = loopir_arms
    assert base_arm.schedule is not None
    assert regblock_arm.schedule is not None
    assert base_arm.schedule.plan.auto_policy is not None
    assert regblock_arm.schedule.plan.auto_policy is not None
    assert base_arm.schedule.plan.auto_policy.regblock_enabled is False
    assert regblock_arm.schedule.plan.auto_policy.regblock_enabled is True
    reconstructed = ops._stitch_regblock_dual_path(
        copy.deepcopy(regblock_arm.llir_function),
        copy.deepcopy(base_arm.llir_function),
        options.scheduler.regblock_max_n,
    )
    assert reconstructed is not None
    assert _lower_to_cpp(reconstructed) == production_cpp


@pytest.mark.parametrize("explicit_update", [False, True])
@pytest.mark.parametrize("regblock_enabled", [False, True])
def test_dual_path_dense_arms_are_migrated_through_reduce_out(
    explicit_update,
    regblock_enabled,
):
    """Both dense-matmul dual arms now lower through the reduce-out family."""

    options = CompileOptions.from_environment(environ={})
    assert (
        ops._build_regblock_dual_path(
            _build_spmm_cin(
                sparse_format="dd",
                explicit_update=explicit_update,
            ),
            None,
            compile_options=options,
        )
        is not None
    )
    arm_options = replace(
        options.with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(),
    )
    kernel = compile_cin_via_loopir(
        _build_spmm_cin(
            sparse_format="dd",
            explicit_update=explicit_update,
        ),
        (4, 6),
        (
            ((4, 5), torch.float32),
            ((5, 6), torch.float32),
        ),
        compile_options=arm_options,
    )
    plan = kernel.schedule.plan
    assert plan.workspace is not None and plan.workspace.dense
    assert plan.auto_policy is not None
    assert plan.auto_policy.regblock_enabled is regblock_enabled
    assert {tile.loop for tile in plan.tiles} >= {
        plan.workspace.reduction_loop,
        plan.workspace.axis_loops[0],
    }


@pytest.mark.parametrize("explicit_update", [False, True])
def test_dual_path_phase6_scope_keeps_hierarchical_ss_open(explicit_update):
    """The hierarchical-compressed ss dual arm remains an open boundary."""

    options = CompileOptions.from_environment(environ={})
    assert (
        ops._build_regblock_dual_path(
            _build_spmm_cin(
                sparse_format="ss",
                explicit_update=explicit_update,
            ),
            None,
            compile_options=options,
        )
        is not None
    )
    arm_options = replace(
        options.with_regblock_enabled(True),
        requested_schedule=Schedule(),
    )
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            _build_spmm_cin(
                sparse_format="ss",
                explicit_update=explicit_update,
            ),
            (4, 6),
            (
                ((4, 5), torch.float32),
                ((5, 6), torch.float32),
            ),
            compile_options=arm_options,
        )
    assert error.value.defect.code == "unsupported_program_shape"
