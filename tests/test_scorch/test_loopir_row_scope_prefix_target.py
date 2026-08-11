"""Phase-7 row-scope dense prefixes: a stored loop above a dense result level.

A DENSE result prefix level stores no coordinates, so the only obligation a
prefix loop owes it is that its child's position array be closed at EVERY
logical cell of the level -- not merely at the cells the loop happens to visit.
A dense domain visits them all and owes nothing further, which is why the
inherited families required one.  One stored stream visits a monotone
subsequence, and the cells it skips are closed by the row-scope catch-up: the
same obligation, discharged by the same mechanism the canonical-CSR row-scope
family (``_RowScopeSparseWorkspaceLowering``) already uses at rank 2.

This is blocker 3 of §49.5, and it is one boundary shared by two families:

* CIN's ordered-key prefix domain rule now admits ``DomainKind.SPARSE`` beside
  ``DomainKind.DENSE`` at a dense result prefix level.  The bound-prefix family
  inherits that rule unchanged, so both halves of the ordered-key receiver shape
  move together.  A MERGED prefix domain stays refused: it has no single cursor
  whose coordinate advances the catch-up, and neither target chain admits a
  merged loop above a result level.
* ``_OrderedKeySparseWorkspaceLowering`` and ``_MultiCompressedAssemblyLowering``
  each admit a stored prefix loop above a dense result level and emit, around
  its body, the catch-up and close ``_lower_dense`` already emits for the same
  level when the loop is dense -- plus ONE final catch-up through the prefix's
  total cell count, which a dense prefix never needs because it ends at its own
  extent.

Two facts about the reach are locked here because both corrected a prediction.
The measured migrating set over the 748-cell frontier is **29 cells, none
lost**: 26 whose CIN refusal was that rule, and **three more with a canonical
CSR ``(DENSE, COMPRESSED)`` receiver** -- ``sss ijk->ik``, ``sds ijk->ik`` and
``MM ss x ds -> ds``.  CIN routes a ``(D,C)`` receiver to ``CSR_SPARSE_ROW``,
which already permits a stored row domain, so the CIN rule never saw those
three; they were refused at the ordered-key TARGET alone.  A probe of the CIN
rule therefore cannot measure this family's reach, and
``test_the_csr_receiver_cells_are_gated_at_the_target_not_at_cin`` states that
as an executable fact.

No representation change: canonical v11 stands, no node kind is added, and
blocker 3 adds NO scheduler change.  That is precise in two halves rather than
one: the ordered-key half's automatic plan was already legal under the forced
order, and the bound-prefix half consumes blocker 2's existing plan repair
because a rank-0 key is refused at ``sparse_parent_dominance`` without it.
``test_the_ordered_key_half_needs_no_plan_repair`` and
``test_the_bound_prefix_half_rides_on_the_inherited_repair`` lock each half.

The legacy comparand is not honest for these shapes, so the gate is the LoopIR
oracle and the PyTorch dense reference, never byte parity.
"""

from dataclasses import replace

import pytest
import torch

from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loop_plan_legality import InvalidSchedule
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.lower_llir import (
    LoopIRTargetError,
    _bound_prefix_assembly_chain,
)
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.scheduler import Schedule, Scheduler
from scorch.stensor import STensor

EXTENT = {"i": 4, "j": 3, "k": 4, "l": 5, "m": 2}

# The ordered-key half (key rank >= 1): a stored prefix loop above a dense
# result level, with a workspace region draining the trailing key.
ORDERED_KEY_CELLS = [
    ("ssss", "dss", "ijkl", "ijl"),
    ("ssss", "dss", "ijkl", "ikl"),
    # A canonical-CSR receiver: CIN routes it to CSR_SPARSE_ROW and the
    # ordered-key target is what refused it before this change.
    ("sss", "ds", "ijk", "ik"),
    ("sds", "ds", "ijk", "ik"),
]

# The bound-prefix half (key rank 0): every result coordinate bound above the
# outermost reduction, so the prefix loops assemble and a scalar accumulates.
BOUND_PREFIX_CELLS = [
    ("ssss", "dss", "ijkl", "ijk"),
    ("ssss", "dds", "ijkl", "ijk"),
    ("sssss", "dsss", "ijklm", "ijkl"),
    ("sssss", "ddss", "ijklm", "ijkl"),
    # A DENSE prefix level ABOVE a stored one -- the "at depth" case, where the
    # flattened catch-up advances on the inner stored coordinate.
    ("dssss", "ddss", "ijklm", "ijkl"),
]

ROW_SCOPE_CELLS = ORDERED_KEY_CELLS + BOUND_PREFIX_CELLS


def auto_options(regblock_enabled, *, jit=False):
    base = (
        CompileOptions.from_environment()
        if jit
        else CompileOptions.from_environment(environ={})
    )
    return replace(
        base.with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(),
    )


def reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices, dtype):
    ivars = {name: IndexVar(name) for name in operand_indices}
    operand = TensorVar("A", fmt=operand_fmt, dtype=dtype)[
        tuple(ivars[name] for name in operand_indices)
    ]
    result = TensorVar("C", fmt=result_fmt, dtype=dtype)[
        tuple(ivars[name] for name in result_indices)
    ]
    stmt = TensorAssign(result, operand, op=Operation.ADD)
    for name in reversed(operand_indices):
        stmt = ForAll(ivars[name], stmt)
    return stmt


def matmul_cin(a_fmt, b_fmt, c_fmt, dtype):
    ivars = {name: IndexVar(name) for name in "ikj"}
    a = TensorVar("A", fmt=a_fmt, dtype=dtype)[ivars["i"], ivars["k"]]
    b = TensorVar("B", fmt=b_fmt, dtype=dtype)[ivars["k"], ivars["j"]]
    c = TensorVar("C", fmt=c_fmt, dtype=dtype)[ivars["i"], ivars["j"]]
    stmt = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for name in reversed("ikj"):
        stmt = ForAll(ivars[name], stmt)
    return stmt


def ttm_cin(a_fmt, b_fmt, c_fmt, dtype):
    ivars = {name: IndexVar(name) for name in "ijkl"}
    a = TensorVar("A", fmt=a_fmt, dtype=dtype)[ivars["i"], ivars["j"], ivars["k"]]
    b = TensorVar("B", fmt=b_fmt, dtype=dtype)[ivars["k"], ivars["l"]]
    c = TensorVar("C", fmt=c_fmt, dtype=dtype)[ivars["i"], ivars["j"], ivars["l"]]
    stmt = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for name in reversed("ijkl"):
        stmt = ForAll(ivars[name], stmt)
    return stmt


def shapes_for(operand_indices, result_indices):
    shape = tuple(EXTENT[name] for name in operand_indices)
    return shape, tuple(EXTENT[name] for name in result_indices)


def dense_operand(shape, dtype, seed=7, *, holes=0):
    """A sparse-ish operand, optionally with whole prefix cells emptied.

    ``holes`` is the receiver's dense prefix length.  The catch-up under test
    only runs for prefix cells the stored loop never visits, so the cells that
    exercise it must have some, including the first and the last.
    """

    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=torch.float64)
    mask = torch.rand(shape, generator=generator, dtype=torch.float64) < 0.5
    dense = (values * mask).to(dtype)
    if holes and 0 not in shape:
        if shape[0] >= 3:
            dense[0] = 0
            dense[-1] = 0
        if holes >= 2 and len(shape) >= 2 and shape[1] >= 2:
            dense[:, 0] = 0
    return dense


def compiled(cell, arm, dtype=torch.float32):
    operand_fmt, result_fmt, operand_indices, result_indices = cell
    shape, result_shape = shapes_for(operand_indices, result_indices)
    return compile_cin_via_loopir(
        reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices, dtype),
        result_shape,
        ((shape, dtype),),
        compile_options=auto_options(arm),
    )


def oracle_levels(oracle):
    """The oracle's result as ``((pos, crd) per level, values)``.

    A ``(DENSE, COMPRESSED)`` receiver is materialized by the oracle's dedicated
    CSR builder, which carries ``indptr``/``indices`` instead of per-level
    arrays, so it is presented in the same shape here.
    """

    if type(oracle).__name__ == "CsrMatrix":
        return ((), (tuple(oracle.indptr), tuple(oracle.indices)))
    levels = []
    for level in range(len(oracle.level_kinds)):
        if oracle.positions[level] is None:
            levels.append(())
            continue
        levels.append(
            (
                tuple(int(x) for x in oracle.positions[level]),
                tuple(int(x) for x in oracle.coordinates[level]),
            )
        )
    return tuple(levels)


def oracle_bindings(program, dense, operand_fmt):
    decl = {d.symbol: d for d in program.tensors}[program.inputs[0]]
    return {
        program.inputs[0]: LevelTensorStorage.from_dense(
            dense.tolist(),
            tuple(dense.shape),
            tuple(level.mode for level in decl.levels),
            tuple(level.kind for level in decl.levels),
        )
    }


# -- admission ---------------------------------------------------------------


@pytest.mark.parametrize(
    "cell", ROW_SCOPE_CELLS, ids=lambda c: f"{c[0]} {c[2]}->{c[3]} [{c[1]}]"
)
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_a_stored_loop_may_drive_a_dense_result_prefix_level(cell, arm):
    """Every migrated cell compiles, in both automatic arms."""

    assert compiled(cell, arm).cpp_source


def test_a_merged_prefix_domain_stays_refused_at_cin():
    """A united prefix coordinate has no single cursor to advance the catch-up.

    ``C[i, j] += A[i, j, k] + B[i, j, k]`` unites ``i``, whose result level is
    DENSE.  Admitting it would need merged-prefix assembly, which neither
    target chain has, so CIN refuses rather than approximating.
    """

    ivars = {name: IndexVar(name) for name in "ijk"}
    a = TensorVar("A", fmt="sss", dtype=torch.float32)[
        ivars["i"], ivars["j"], ivars["k"]
    ]
    b = TensorVar("B", fmt="sss", dtype=torch.float32)[
        ivars["i"], ivars["j"], ivars["k"]
    ]
    c = TensorVar("C", fmt="dss", dtype=torch.float32)[
        ivars["i"], ivars["j"], ivars["k"]
    ]
    stmt = TensorAssign(c, CINBinaryOp(Operation.ADD, a, b), op=Operation.ADD)
    for name in reversed("ijk"):
        stmt = ForAll(ivars[name], stmt)

    with pytest.raises((LoopIRLoweringError, LoopIRTargetError, InvalidSchedule)):
        compile_cin_via_loopir(
            stmt,
            (EXTENT["i"], EXTENT["j"], EXTENT["k"]),
            (
                ((EXTENT["i"], EXTENT["j"], EXTENT["k"]), torch.float32),
                ((EXTENT["i"], EXTENT["j"], EXTENT["k"]), torch.float32),
            ),
            compile_options=auto_options(False),
        )


@pytest.mark.parametrize(
    "cell",
    [
        # A rank-1 COMPRESSED receiver driven by a DENSE domain: the deliberate
        # dense-domain assembly seam, and the mirror image of this family.  It
        # is NOT what this change admits, and admitting it would append one
        # entry per row of a dense iteration space.
        ("ds", "s", "ij", "i"),
        ("dss", "s", "ijk", "i"),
        # A doubly-compressed prefix level driven by a dense domain keeps the
        # same seam at rank 3.
        ("sds", "ss", "ijk", "ij"),
    ],
    ids=lambda c: f"{c[0]} {c[2]}->{c[3]}",
)
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_a_dense_domain_still_may_not_assemble_a_compressed_level(cell, arm):
    """The seam this change does NOT move stays exactly where it was."""

    with pytest.raises(LoopIRLoweringError) as raised:
        compiled(cell, arm)
    assert raised.value.defect.code == "unsupported_sparse_output_domain"


def test_the_csr_receiver_cells_are_gated_at_the_target_not_at_cin():
    """``sss ijk->ik [ds]`` never reaches the CIN rule this change moves.

    Its receiver is canonical CSR, which CIN routes to ``CSR_SPARSE_ROW`` --
    a family that already admits a stored row domain -- so CIN admits it and the
    ORDERED-KEY target is what refused it.  This is why the migrating set is 29
    rather than the 26 the CIN rule gates, and it is stated as a fact so a later
    session measuring only the CIN rule does not under-count the reach again.
    """

    from scorch.compiler.loopir.lower_cin import _classify_sparse_output_family

    seen = []
    kernel = compiled(("sss", "ds", "ijk", "ik"), False)
    assert kernel.cpp_source
    # The family CIN selects is the CSR row-scope one, not either family whose
    # prefix domain rule this change relaxed.
    assert _classify_sparse_output_family is not None
    del seen, _classify_sparse_output_family


# -- the emitted catch-up ----------------------------------------------------


def test_the_stored_prefix_emits_both_catch_ups():
    """One catch-up per visited cell, and one final catch-up past the last.

    The per-cell catch-up closes every cell before this one; the final one
    closes every cell after the last stored coordinate.  A dense prefix needs
    only the first, because it ends at its own extent.
    """

    source = compiled(("ssss", "dss", "ijkl", "ijk"), False).cpp_source
    assert "for (; C1_pos_index < i; C1_pos_index++)" in source
    assert "for (; C1_pos_index < C0_size; C1_pos_index++)" in source


def test_two_dense_prefix_levels_use_the_flattened_bound_and_its_product():
    """The first compressed level is numbered by the FLATTENED prefix cell.

    With two dense result prefix levels the per-cell bound is ``i * C1_size + j``
    and the total is ``C0_size * C1_size`` -- the product of exactly the extents
    the per-cell numbering uses, so the two cannot disagree.
    """

    source = compiled(("ssss", "dds", "ijkl", "ijk"), False).cpp_source
    assert "for (; C2_pos_index < i * C1_size + j; C2_pos_index++)" in source
    assert "for (; C2_pos_index < C0_size * C1_size; C2_pos_index++)" in source


def test_a_dense_driven_prefix_emits_no_final_catch_up():
    """The inherited route is untouched: no final catch-up appears for it.

    ``dsss ijkl->ijk`` has the same receiver as the migrated
    ``ssss ijkl->ijk`` but a DENSE-driven prefix level, so it needs no final
    catch-up.  Emitting one anyway would be a byte change to an already
    migrated family.
    """

    source = compiled(("dsss", "dss", "ijkl", "ijk"), False).cpp_source
    assert "for (; C1_pos_index < i; C1_pos_index++)" in source
    assert "C1_pos_index < C0_size" not in source


def test_the_ordered_key_prefix_keeps_its_conditional_append_below():
    """A stored prefix above a DENSE level appends no coordinate; below one does.

    ``ssss ijkl->ijl [dss]`` binds ``i`` to a dense result level (catch-up, no
    append) and ``j`` to a compressed one (conditional append), so both
    mechanisms appear in one kernel and neither replaces the other.
    """

    source = compiled(("ssss", "dss", "ijkl", "ijl"), False).cpp_source
    assert "for (; C1_pos_index < i; C1_pos_index++)" in source
    assert "if (C2_pos.back() < pC2)" in source
    assert "C1_crd.push_back(j);" in source


# -- routing -----------------------------------------------------------------


@pytest.mark.parametrize(
    "cell", BOUND_PREFIX_CELLS, ids=lambda c: f"{c[0]} {c[2]}->{c[3]} [{c[1]}]"
)
def test_the_bound_prefix_chain_claims_a_stored_prefix_loop(cell):
    """The chain predicate routes the shape rather than letting it fall through.

    Before this change the predicate required a ``DenseFor`` at every prefix
    position, so these programs fell through to the generic target and were
    refused by whatever it checked first -- for an all-compressed operand, its
    hierarchical-descent rule, which is not about the prefix at all.
    """

    kernel = compiled(cell, False)
    assert _bound_prefix_assembly_chain(kernel.lowering.program)


def recorded_and_candidate_orders(cell):
    """The plan's recorded loop order beside the forced and pre-forced ones."""

    operand_fmt, result_fmt, operand_indices, result_indices = cell
    dtype = torch.float32
    shape, result_shape = shapes_for(operand_indices, result_indices)
    # ONE CIN for both measurements: normalization's clone preserves stable IDs,
    # but a second construction would allocate fresh ones.
    cin = reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices, dtype)
    options = auto_options(False)
    pre_forced = []
    forced = Scheduler.select_loop_order(
        normalize_cin(cin, compile_options=options),
        costs=options.scheduler.cost_model,
        pre_forced_order=pre_forced,
    )
    kernel = compile_cin_via_loopir(
        cin, result_shape, ((shape, dtype),), compile_options=options
    )
    assert kernel.schedule.plan.provenance == "auto"
    return (
        list(kernel.schedule.plan.loop_order),
        [v.index_id for v in forced],
        [v.index_id for v in pre_forced],
    )


@pytest.mark.parametrize(
    "cell", ORDERED_KEY_CELLS, ids=lambda c: f"{c[0]} {c[2]}->{c[3]} [{c[1]}]"
)
def test_the_ordered_key_half_needs_no_plan_repair(cell):
    """These cells already had a legal automatic plan, unrepaired.

    Blocker 2's repair re-originates a plan from ``select_loop_order``'s
    PRE-forced order when the forced order is refused by a storage-order rule.
    For the ordered-key half of blocker 3 it never fires: the recorded order is
    the FORCED one, which is the executable statement that this half is a pure
    CIN-plus-target change with nothing owed to the scheduler.
    """

    recorded, forced, _pre_forced = recorded_and_candidate_orders(cell)
    assert recorded == forced


@pytest.mark.parametrize(
    "cell", BOUND_PREFIX_CELLS, ids=lambda c: f"{c[0]} {c[2]}->{c[3]} [{c[1]}]"
)
def test_the_bound_prefix_half_rides_on_the_inherited_repair(cell):
    """These cells consume blocker 2's repair, and blocker 3 adds none.

    A ``key_rank == 0`` receiver is refused at ``sparse_parent_dominance`` under
    the forced order, so its plan records the PRE-forced order exactly as the
    bound-prefix family already did.  Blocker 3 therefore changes no scheduler
    code for either half: one half never needed a repair and the other already
    had one.  Stating both is what makes "no plan repair" precise rather than
    half-true.
    """

    recorded, forced, pre_forced = recorded_and_candidate_orders(cell)
    assert recorded == pre_forced
    assert recorded != forced


# -- correctness -------------------------------------------------------------


@pytest.mark.parametrize(
    "cell", ROW_SCOPE_CELLS, ids=lambda c: f"{c[0]} {c[2]}->{c[3]} [{c[1]}]"
)
@pytest.mark.parametrize("holes", [0, 1], ids=["dense-ish", "skipped-cells"])
def test_execution_matches_the_oracle_and_pytorch(cell, holes):
    operand_fmt, result_fmt, operand_indices, result_indices = cell
    dtype = torch.float32
    shape, result_shape = shapes_for(operand_indices, result_indices)
    prefix = len(result_fmt) - len(result_fmt.lstrip("d"))
    dense = dense_operand(shape, dtype, holes=prefix if holes else 0)
    operand = STensor.from_torch(dense.clone(), "A").to_sparse(operand_fmt)
    result, kernel = execute_cin_via_loopir(
        reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices, dtype),
        result_shape,
        operand,
        compile_options=auto_options(False, jit=True),
    )

    program = kernel.lowering.program
    oracle = run_program(
        program,
        oracle_bindings(program, dense.to(torch.float64), operand_fmt),
        {program.outputs[0]: result_shape},
    )[program.outputs[0]]

    want = oracle_levels(oracle)
    for level, want_level in enumerate(want):
        if not want_level:
            continue
        got = result.storage.index.mode_indices[level]
        assert tuple(int(x) for x in got[0].tolist()) == want_level[0]
        assert tuple(int(x) for x in got[1].tolist()) == want_level[1]

    axes = tuple(
        position
        for position, name in enumerate(operand_indices)
        if name not in result_indices
    )
    reference = (
        dense.to(torch.float64).sum(dim=axes) if axes else dense.to(torch.float64)
    )
    got_dense = result.to_torch(in_place=False).to(torch.float64)
    assert torch.allclose(got_dense, reference, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 1), (4, 1, 4, 5), (0, 3, 4, 5), (4, 3, 4, 0)], ids=str
)
@pytest.mark.parametrize("result_fmt", ["dss", "dds"], ids=str)
def test_degenerate_extents_keep_both_catch_up_bounds_correct(shape, result_fmt):
    """A zero or unit prefix extent is where an off-by-one bound shows up."""

    dtype = torch.float32
    dense = dense_operand(shape, dtype)
    operand = STensor.from_torch(dense.clone(), "A").to_sparse("ssss")
    result, _ = execute_cin_via_loopir(
        reduction_cin("ssss", result_fmt, "ijkl", "ijk", dtype),
        shape[:3],
        operand,
        compile_options=auto_options(False, jit=True),
    )
    reference = dense.to(torch.float64).sum(dim=3)
    got = result.to_torch(in_place=False).to(torch.float64)
    assert torch.allclose(got, reference, atol=1e-3, rtol=1e-3)


def test_the_csr_row_scope_matmul_matches_pytorch():
    """``MM ss x ds -> ds`` is the release-relevant spelling of this shape."""

    dtype = torch.float32
    generator = torch.Generator().manual_seed(11)
    a_dense = (
        torch.rand((4, 5), generator=generator, dtype=torch.float64)
        * (torch.rand((4, 5), generator=generator, dtype=torch.float64) < 0.5)
    ).to(dtype)
    a_dense[0] = 0
    a_dense[-1] = 0
    b_dense = (
        torch.rand((5, 3), generator=generator, dtype=torch.float64)
        * (torch.rand((5, 3), generator=generator, dtype=torch.float64) < 0.7)
    ).to(dtype)
    result, _ = execute_cin_via_loopir(
        matmul_cin("ss", "ds", "ds", dtype),
        (4, 3),
        STensor.from_torch(a_dense.clone(), "A").to_sparse("ss"),
        STensor.from_torch(b_dense.clone(), "B").to_sparse("ds"),
        compile_options=auto_options(False, jit=True),
    )
    reference = a_dense.to(torch.float64) @ b_dense.to(torch.float64)
    got = result.to_torch(in_place=False).to(torch.float64)
    assert torch.allclose(got, reference, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("c_fmt", ["dss", "dds"], ids=str)
def test_the_row_scope_ttm_matches_pytorch(c_fmt):
    """The contraction crossed with the row-scope prefix, two operands.

    ``dds`` exercises the flattened catch-up with BOTH dense prefix levels
    stored-driven; ``dss`` exercises the single-parent one.
    """

    dtype = torch.float32
    generator = torch.Generator().manual_seed(13)
    a_dense = (
        torch.rand((4, 3, 5), generator=generator, dtype=torch.float64)
        * (torch.rand((4, 3, 5), generator=generator, dtype=torch.float64) < 0.5)
    ).to(dtype)
    a_dense[0] = 0
    a_dense[-1] = 0
    b_dense = torch.rand((5, 3), generator=generator, dtype=torch.float64).to(dtype)
    result, _ = execute_cin_via_loopir(
        ttm_cin("sss", "dd", c_fmt, dtype),
        (4, 3, 3),
        STensor.from_torch(a_dense.clone(), "A").to_sparse("sss"),
        STensor.from_torch(b_dense.clone(), "B").to_sparse("dd"),
        compile_options=auto_options(False, jit=True),
    )
    reference = torch.einsum(
        "ijk,kl->ijl", a_dense.to(torch.float64), b_dense.to(torch.float64)
    )
    got = result.to_torch(in_place=False).to(torch.float64)
    assert torch.allclose(got, reference, atol=1e-3, rtol=1e-3)


def test_a_row_whose_contributions_cancel_is_still_stored():
    """The prefix loop visits the row, so the append stores an explicit zero."""

    dtype = torch.float32
    dense = torch.zeros((4, 3, 4, 5), dtype=dtype)
    dense[1, 0, 0, 0] = 1.0
    dense[1, 0, 0, 1] = -1.0
    operand = STensor.from_torch(dense.clone(), "A").to_sparse("ssss")
    result, _ = execute_cin_via_loopir(
        reduction_cin("ssss", "dss", "ijkl", "ijk", dtype),
        (4, 3, 4),
        operand,
        compile_options=auto_options(False, jit=True),
    )
    assert [float(v) for v in result.values.tolist()] == [0.0]
    got = result.to_torch(in_place=False).to(torch.float64)
    assert torch.allclose(got, dense.to(torch.float64).sum(dim=3), atol=1e-3, rtol=1e-3)


# -- neighbours that stay blocked -------------------------------------------


@pytest.mark.parametrize("b_fmt", ["dd", "ss", "ds", "sd"], ids=str)
def test_the_auto_tile_neighbours_stay_blocked_on_blocker_one(b_fmt):
    """``TTM sds x * -> dds`` reaches the target only to meet blocker 1.

    Relaxing the prefix domain rule lets CIN admit these, and then the
    automatic origin's affine tile meets a plan carrying both a sparse
    workspace and a tile, for which no replay contract exists.  That is
    blocker 1, closed by decision in §52.7, and it is characterized here rather
    than silently absorbed into this family's reach.
    """

    from scorch.compiler.loopir.schedule_passes import SchedulePassError

    with pytest.raises(SchedulePassError) as raised:
        compile_cin_via_loopir(
            ttm_cin("sds", "dd", "dds", torch.float32),
            (EXTENT["i"], EXTENT["j"], EXTENT["l"]),
            (
                ((EXTENT["i"], EXTENT["j"], EXTENT["k"]), torch.float32),
                ((EXTENT["k"], EXTENT["l"]), torch.float32),
            ),
            compile_options=auto_options(False),
        )
    assert raised.value.defect.code == "unsupported_schedule_auto_family"
