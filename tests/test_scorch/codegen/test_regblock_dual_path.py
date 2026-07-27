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
from scorch.compiler.loopir.schedule_passes import SchedulePassError
from scorch.compiler.scheduler import (
    Schedule,
    Scheduler,
    regblock_force,
    _regblock_max_n,
    _regblock_tile_width,
)


def _build_spmm_cin(*, sparse_format="ds", bind_shapes=False):
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
    assignment = TensorAssign(
        c[i, k],
        a[i, j] * b[j, k],
        op=Operation.ADD,
    )
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


def test_dual_path_composes_exactly_the_two_verified_arm_lowerings():
    """The production stitch is target-level composition, not a third lowering.

    Phase-6 ownership decision: each regblock arm is an independently
    verified automatic schedule (the tile-free contract and the stack-form
    contract, both proven byte-identical through LoopIR), and the dual
    kernel is exactly the register-block arm's lowering with its single
    top-level compute loop replaced by a runtime free-dim branch over the
    two arms' compute loops.  Reconstructing that stitch from the two arm
    lowerings must reproduce the production builder byte-for-byte; the
    runtime branch itself is target-lowering state and its migration
    belongs to Phase 7.
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
                _build_spmm_cin(),
                (4, 6),
                input_bindings,
                compile_options=arm_options,
            )
        )

    built = ops._build_regblock_dual_path(
        _build_spmm_cin(bind_shapes=True),
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


@pytest.mark.parametrize(
    ("left_format", "error_type", "error_code"),
    [
        ("dd", SchedulePassError, "unsupported_schedule_auto_family"),
        ("ss", LoopIRTargetError, "unsupported_program_shape"),
    ],
)
def test_dual_path_phase6_scope_keeps_unmigrated_arms_open(
    left_format,
    error_type,
    error_code,
):
    """The production dual helper reaches families Phase 6 has not migrated."""

    options = CompileOptions.from_environment(environ={})
    assert (
        ops._build_regblock_dual_path(
            _build_spmm_cin(sparse_format=left_format),
            None,
            compile_options=options,
        )
        is not None
    )
    arm_options = replace(
        options.with_regblock_enabled(True),
        requested_schedule=Schedule(),
    )
    with pytest.raises(error_type) as error:
        compile_cin_via_loopir(
            _build_spmm_cin(sparse_format=left_format),
            (4, 6),
            (
                ((4, 5), torch.float32),
                ((5, 6), torch.float32),
            ),
            compile_options=arm_options,
        )
    assert error.value.defect.code == error_code
