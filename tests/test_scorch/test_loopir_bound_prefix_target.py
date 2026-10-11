"""Phase-7 bound-prefix accumulation: the rank-0 ordered key (blocker 2).

``_ordered_key_split`` splits one ordered sparse reduction's result
coordinates into a prefix bound above the outermost reduction and a key bound
below the innermost one.  A ``key_rank == 0`` split -- every result coordinate
bound above every reduction -- is a legitimate split, not a degenerate
instance of a rank->=1 one: the prefix loops already visit result cells in
lexicographic order, so the reduction is a scalar accumulation under an
already-complete result prefix and no workspace region participates.  Neither
workspace node can express such an accumulator (``WorkspaceDecl`` is bound to a
``TileId``, ``SparseWorkspaceDecl`` requires one or more key dimensions), so it
is an LLIR local exactly as every other family's reduction accumulator is, and
this family adds no node kind and no canonical schema change of its own (the
schema later moved to v12 for the program's assembly-strategy fact).

Two changes gate together, and this module locks both halves and the gating:

* CIN admits the rank-0 split as ``BOUND_PREFIX_ACCUMULATION``, keeping the
  ordered-key prefix domain rules unchanged, and lowers it to the same
  semantic ``StoreReduce`` leaf every sparse reduction family uses.
* ``_MultiCompressedAssemblyLowering`` -- which already admits these receivers
  and already emits the per-level append, the conditional compressed-parent
  append with child position close, the dense-prefix catch-up and the root
  close -- grows an optional reduction sub-nest below its innermost assembly
  loop whose leaf accumulates into a scalar the ordered append reads.
* ``Scheduler``'s plan origin repairs the one unchecked block of
  ``select_loop_order``: the forced reorder that moves the last free variable
  inward.  For a bound-prefix program that reorder puts a compressed physical
  parent below its child, and the plan's own legality rules refuse it.

Neither half migrates anything alone, which
``test_the_two_halves_gate_together`` states as an executable fact.

The legacy comparand is not honest for this family -- the generic legacy route
writes an unsized result vector for a sparse-result reduction, which is why
``_validate_loop_kinds`` refuses the unscheduled form outright -- so the gate is
the LoopIR oracle and the PyTorch dense reference, never byte parity.
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
from scorch.compiler.identity import IndexId
from scorch.compiler.loop_plan_legality import InvalidSchedule
from scorch.compiler.loopir.levels import LevelTensorStorage
from scorch.compiler.loopir.lower_cin import (
    LoopIRLoweringError,
    _ordered_key_split,
    _SparseOutputReduction,
    lower_normalized_cin_to_loopir,
)
from scorch.compiler.loopir.lower_llir import (
    LoopIRTargetError,
    _bound_prefix_assembly_chain,
    _multi_compressed_assembly_chain,
)
from scorch.compiler.loopir.oracle import run_program
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.scheduler import Schedule, Scheduler
from scorch.format import LevelType
from scorch.stensor import STensor

EXTENT = {"i": 3, "j": 4, "k": 5, "l": 2}

# Every cell the 748-cell frontier measures as newly ADMITTED, in both
# automatic arms: the whole reach of the rank-0 split under the ordered-key
# prefix domain rules.
MIGRATED = [
    ("ss", "s", "ij", "i"),
    ("sd", "s", "ij", "i"),
    ("sss", "s", "ijk", "i"),
    ("ssd", "s", "ijk", "i"),
    ("sds", "s", "ijk", "i"),
    ("ssss", "s", "ijkl", "i"),
    ("ssss", "sss", "ijkl", "ijk"),
    ("sdss", "s", "ijkl", "i"),
]

# Cells admitted once a compressed result level may be driven by a dense
# domain, and once the doubly-compressed and canonical-CSR receivers take this
# family whenever no key is drained (2026-10-11).  Each was a refused neighbour
# below until then, and each is arm-invariant.  A dense-driven leaf level
# appends exactly when its reduction sub-nest ran, so ``ds ij->i [s]`` stores
# only the rows that hold an entry, and the oracle comparison below proves that
# storage cell by cell.
WIDENED = [
    ("ds", "s", "ij", "i"),
    ("dss", "s", "ijk", "i"),
    ("dds", "s", "ijk", "i"),
    ("dsd", "s", "ijk", "i"),
    ("dsss", "sss", "ijkl", "ijk"),
    ("sss", "ss", "ijk", "ij"),
    ("ssd", "ss", "ijk", "ij"),
    ("dss", "ss", "ijk", "ij"),
    ("dds", "ss", "ijk", "ij"),
    ("dsd", "ss", "ijk", "ij"),
    ("sds", "ss", "ijk", "ij"),
    ("sss", "ds", "ijk", "ij"),
    ("ssd", "ds", "ijk", "ij"),
]
ADMITTED = MIGRATED + WIDENED


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


def declared_options(loop_order):
    return CompileOptions.from_environment(
        environ={}, requested_schedule=Schedule(loop_order=tuple(loop_order))
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


def shapes_for(operand_indices, result_indices):
    shape = tuple(EXTENT[name] for name in operand_indices)
    return shape, tuple(EXTENT[name] for name in result_indices)


def dense_operand(shape, dtype, seed=7):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator, dtype=torch.float64)
    mask = torch.rand(shape, generator=generator, dtype=torch.float64) < 0.5
    return (values * mask).to(dtype)


def compiled(cell, arm, dtype=torch.float32, options=None):
    operand_fmt, result_fmt, operand_indices, result_indices = cell
    shape, result_shape = shapes_for(operand_indices, result_indices)
    return compile_cin_via_loopir(
        reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices, dtype),
        result_shape,
        ((shape, dtype),),
        compile_options=auto_options(arm) if options is None else options,
    )


def oracle_bindings(program, dense, operand_fmt):
    decl = {d.symbol: d for d in program.tensors}[program.inputs[0]]
    kinds = tuple(level.kind for level in decl.levels)
    return {
        program.inputs[0]: LevelTensorStorage.from_dense(
            dense.tolist(),
            tuple(dense.shape),
            tuple(level.mode for level in decl.levels),
            kinds,
        )
    }


# -- the split itself --------------------------------------------------------


def test_rank_zero_key_is_a_split_and_so_is_an_interleaved_key():
    """The key is every result coordinate below the outermost reduction."""

    i, j, k = IndexId(1), IndexId(2), IndexId(3)

    # ``C[i] += A[i, j]``: one result coordinate above the only reduction.
    assert _ordered_key_split((LevelType.COMPRESSED,), (i,), (j,), {i: 0, j: 1}) == (
        1,
        0,
    )
    # ``C[j] += A[i, j, k]`` under ``ijk``: the result coordinate sits between
    # reductions at 0 and 2.  It is a key of the region that starts at the
    # outermost reduction, not a refusal and not a bound prefix.
    assert _ordered_key_split(
        (LevelType.COMPRESSED,), (j,), (i, k), {i: 0, j: 1, k: 2}
    ) == (0, 1)
    # A key below the innermost reduction splits exactly as before.
    assert _ordered_key_split(
        (LevelType.COMPRESSED,), (k,), (i, j), {i: 0, j: 1, k: 2}
    ) == (0, 1)
    # What still has no split: result modes bound out of result level order.
    assert (
        _ordered_key_split(
            (LevelType.COMPRESSED, LevelType.COMPRESSED),
            (k, j),
            (i,),
            {i: 0, j: 1, k: 2},
        )
        is None
    )


def test_the_family_is_its_own_member_not_an_ordered_key_workspace():
    """A rank-0 key selects a different family, not a degenerate instance."""

    assert (
        _SparseOutputReduction.BOUND_PREFIX_ACCUMULATION
        is not _SparseOutputReduction.ORDERED_KEY_WORKSPACE
    )


# -- routing -----------------------------------------------------------------


@pytest.mark.parametrize("cell", ADMITTED, ids=lambda c: f"{c[0]} {c[2]}->{c[3]} [{c[1]}]")
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_routing_claims_the_family_for_the_assembly_target(cell, arm):
    kernel = compiled(cell, arm)
    program = kernel.lowering.program
    assert _bound_prefix_assembly_chain(program)
    # The two chain predicates are disjoint: one requires an append leaf, the
    # other an accumulating one.
    assert not _multi_compressed_assembly_chain(program)


@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_the_emitted_kernel_accumulates_into_one_scalar_local(arm):
    kernel = compiled(("ss", "s", "ij", "i"), arm)
    source = kernel.cpp_source
    assert "// Initialize the reduction accumulator" in source
    assert "float C_reduction = 0.0;" in source
    assert "C_reduction += " in source
    assert "C_values.emplace_back(C_reduction);" in source
    # No workspace of either representation participates.
    assert "workspace" not in source.lower()


def test_the_multi_level_receiver_keeps_the_per_level_assembly():
    """A rank-3 receiver still closes every structural level conditionally."""

    kernel = compiled(("ssss", "sss", "ijkl", "ijk"), False)
    source = kernel.cpp_source
    assert source.count("// Assembly compressed _level indices") == 3
    assert "if (C2_pos.back() < pC2)" in source
    assert "if (C1_pos.back() < pC1)" in source
    assert "scorch_vector_set(C2_pos, C1_crd.size(), C2_crd.size());" in source


@pytest.mark.parametrize("cell", MIGRATED, ids=lambda c: f"{c[0]} {c[2]}->{c[3]}")
def test_the_legacy_comparand_still_refuses_this_family(cell):
    """The legacy route cannot lower these programs, before or after the repair.

    ``legacy_generated_cpp`` with an empty ``Schedule()`` is the one
    release-visible surface that consumes the repaired plan through the LEGACY
    lowering, so it is locked here rather than left to be discovered.  It
    refused these shapes before the repair (at the plan boundary, because the
    forced order was illegal) and refuses them after it (inside the legacy
    lowering, because a sparse-result reduction with every free variable above
    the reductions has no legacy form) -- so no program that produced legacy C++
    produces different C++, which is the neutrality property that matters.

    The refusal inside the legacy lowerer used to escape as an unstructured
    ``ValueError`` from ``list.index``.  It is now the structured
    ``unscheduled_sparse_result_reduction`` refusal, so the kind of refusal no
    longer degrades on this surface: it was a structured ``InvalidSchedule``
    before the repair and is a structured ``UnsupportedFeature`` after it.
    """

    from scorch.compiler.diagnostics import UnsupportedFeature
    from scorch.compiler.loopir.pipeline import legacy_generated_cpp

    operand_fmt, result_fmt, operand_indices, result_indices = cell
    shape, result_shape = shapes_for(operand_indices, result_indices)
    with pytest.raises(UnsupportedFeature) as error:
        legacy_generated_cpp(
            reduction_cin(
                operand_fmt, result_fmt, operand_indices, result_indices, torch.float32
            ),
            result_shape,
            ((shape, torch.float32),),
            compile_options=auto_options(False),
        )
    assert [d.code for d in error.value.diagnostics] == [
        "unscheduled_sparse_result_reduction"
    ]


def test_the_doubly_compressed_receiver_is_claimed_exactly_when_no_key_is_drained():
    """``(C, C)`` joins this family without a drained key, and only then.

    With every result coordinate bound above the reduction there is nothing for
    the doubly-compressed workspace family to drain, so the program is this
    family's.  A coordinate bound below the reduction is a drained key: its loop
    is not an assembly loop, the leaf's coordinates are not the first loops'
    own, and routing declines it by binding -- not by layout.
    """

    key_free = lower_normalized_cin_to_loopir(
        normalize_cin(
            reduction_cin("sss", "ss", "ijk", "ij", torch.float32),
            compile_options=declared_options("ijk"),
        )
    ).program
    assert _bound_prefix_assembly_chain(key_free)
    compiled(("sss", "ss", "ijk", "ij"), False)

    drained = lower_normalized_cin_to_loopir(
        normalize_cin(
            reduction_cin("sss", "ss", "ijk", "ik", torch.float32),
            compile_options=declared_options("ijk"),
        )
    ).program
    assert not _bound_prefix_assembly_chain(drained)


def test_routing_reads_bindings_not_positions_for_canonical_csr():
    """A reduction enclosing the column loop is not read as the column.

    Canonical CSR's ``i, k, j`` nest before its workspace rewrite puts a single
    cursor loop in the second position, where this family expects the column.
    Counting loops would claim it; the leaf's coordinates ``(i, j)`` are not the
    first two loops' ``(i, k)``, so routing declines it.
    """

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    left = TensorVar("A", fmt="ds", dtype=torch.float32)[i, k]
    right = TensorVar("B", fmt="ds", dtype=torch.float32)[k, j]
    result = TensorVar("C", fmt="ds", dtype=torch.float32)[i, j]
    cin = ForAll(
        i,
        ForAll(
            k,
            ForAll(
                j,
                TensorAssign(
                    result, CINBinaryOp(Operation.MUL, left, right), op=Operation.ADD
                ),
            ),
        ),
    )
    program = lower_normalized_cin_to_loopir(
        normalize_cin(cin, compile_options=declared_options("ikj"))
    ).program
    assert not _bound_prefix_assembly_chain(program)


# -- correctness -------------------------------------------------------------


@pytest.mark.parametrize("cell", ADMITTED, ids=lambda c: f"{c[0]} {c[2]}->{c[3]} [{c[1]}]")
@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
def test_execution_matches_the_oracle_and_pytorch(cell, arm, dtype):
    operand_fmt, result_fmt, operand_indices, result_indices = cell
    shape, result_shape = shapes_for(operand_indices, result_indices)
    dense = dense_operand(shape, dtype)
    operand = STensor.from_torch(dense.clone(), "A").to_sparse(operand_fmt)
    result, kernel = execute_cin_via_loopir(
        reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices, dtype),
        result_shape,
        operand,
        compile_options=auto_options(arm, jit=True),
    )

    program = kernel.lowering.program
    oracle = run_program(
        program,
        oracle_bindings(program, dense.to(torch.float64), operand_fmt),
        {program.outputs[0]: result_shape},
    )[program.outputs[0]]

    # Exact (pos, crd) storage against the oracle, level by level.
    for level in range(len(oracle.level_kinds)):
        if oracle.positions[level] is None:
            continue
        got = result.storage.index.mode_indices[level]
        assert [int(x) for x in got[0].tolist()] == [
            int(x) for x in oracle.positions[level]
        ]
        assert [int(x) for x in got[1].tolist()] == [
            int(x) for x in oracle.coordinates[level]
        ]
    assert len(result.values.tolist()) == len(oracle.values)

    axes = tuple(
        position
        for position, name in enumerate(operand_indices)
        if name not in result_indices
    )
    reference = dense.to(torch.float64).sum(dim=axes)
    got_dense = result.to_torch(in_place=False).to(torch.float64)
    assert torch.allclose(got_dense, reference, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 1, 5), (0, 4, 5), (3, 4, 0)], ids=str)
def test_degenerate_extents_keep_the_family_correct(shape):
    dtype = torch.float32
    dense = dense_operand(shape, dtype)
    operand = STensor.from_torch(dense.clone(), "A").to_sparse("sss")
    result, _ = execute_cin_via_loopir(
        reduction_cin("sss", "s", "ijk", "i", dtype),
        (shape[0],),
        operand,
        compile_options=auto_options(False, jit=True),
    )
    reference = dense.to(torch.float64).sum(dim=(1, 2))
    got = result.to_torch(in_place=False).to(torch.float64)
    assert torch.allclose(got, reference, atol=1e-3, rtol=1e-3)


def test_exact_cancellation_still_stores_its_entry():
    """A row whose contributions cancel is stored, with the value zero.

    The prefix loop visits the row, so the append is unconditional: the family
    stores an explicit zero rather than dropping a structurally present row.
    """

    dtype = torch.float64
    dense = torch.zeros((3, 4), dtype=dtype)
    dense[0, 0] = 1.5
    dense[0, 1] = -1.5
    dense[2, 3] = 2.0
    operand = STensor.from_torch(dense.clone(), "A").to_sparse("ss")
    result, _ = execute_cin_via_loopir(
        reduction_cin("ss", "s", "ij", "i", dtype),
        (3,),
        operand,
        compile_options=auto_options(False, jit=True),
    )
    stored = result.storage.index.mode_indices[0]
    # Rows 0 and 2 are stored; row 1 has no stored entry at all.
    assert [int(x) for x in stored[1].tolist()] == [0, 2]
    assert result.values.tolist() == pytest.approx([0.0, 2.0])
    assert torch.allclose(
        result.to_torch(in_place=False).to(dtype),
        dense.sum(dim=1),
        atol=1e-9,
        rtol=1e-9,
    )


# -- the refused neighbours, each at its own code ----------------------------


# The single-operand refused neighbours this section used to lock have all
# moved, and each is proven where it went rather than locked here:
#
# - a COMPRESSED result prefix level driven by a DENSE domain (``ds ij->i``,
#   ``dss ijk->i``, ``dsss ijkl->ijk``) is ``WIDENED`` above;
# - a DENSE result prefix level driven by a stored stream is migrated by
#   blocker 3 -- ``test_loopir_row_scope_prefix_target.py`` owns that cell;
# - a result coordinate interleaved between two reductions (``sss ijk->j``) is
#   a key of the ordered-key workspace region since 2026-10-11 --
#   ``test_loopir_interleaved_workspace_key.py`` executes it.
#
# The residue of the first two rules is a prefix level driven by a MERGED
# domain, which needs two operands; it is locked by
# ``test_a_merged_prefix_domain_stays_refused_at_cin`` in the row-scope module.


@pytest.mark.parametrize("arm", [False, True], ids=["direct", "regblock"])
def test_a_merged_reduction_domain_is_refused_at_cin(arm):
    """``C[i] += A[i, j] * B[j]`` intersects on the reduction coordinate."""

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("A", fmt="ss", dtype=torch.float32)[i, j]
    b = TensorVar("B", fmt="s", dtype=torch.float32)[j]
    c = TensorVar("C", fmt="s", dtype=torch.float32)[i]
    cin = ForAll(
        i,
        ForAll(j, TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)),
    )
    with pytest.raises(LoopIRLoweringError) as error:
        compile_cin_via_loopir(
            cin,
            (EXTENT["i"],),
            (
                ((EXTENT["i"], EXTENT["j"]), torch.float32),
                ((EXTENT["j"],), torch.float32),
            ),
            compile_options=auto_options(arm),
        )
    assert error.value.defect.code == "unsupported_merged_reduction"


def test_the_generated_accumulator_identifier_is_name_reserved():
    """An input spelled ``C_reduction`` fails closed, it does not shadow.

    The accumulator's C++ identifier goes through the same
    ``_reserve_generated_name`` authority as every merge temporary, so a user
    tensor that happens to spell it collides at the reservation rather than
    silently aliasing the accumulator inside the emitted kernel.  §53.11
    measured this and left it untested; it is a unit test now.
    """

    i, j = IndexVar("i"), IndexVar("j")
    a = TensorVar("C_reduction", fmt="ss", dtype=torch.float32)[i, j]
    c = TensorVar("C", fmt="s", dtype=torch.float32)[i]
    cin = ForAll(i, ForAll(j, TensorAssign(c, a, op=Operation.ADD)))
    with pytest.raises(LoopIRTargetError) as error:
        compile_cin_via_loopir(
            cin,
            (EXTENT["i"],),
            (((EXTENT["i"], EXTENT["j"]), torch.float32),),
            compile_options=auto_options(False),
        )
    assert error.value.defect.code == "generated_name_collision"
    assert "C_reduction" in error.value.defect.message


@pytest.mark.parametrize(
    "cell,code",
    [
        # ``forced`` moves the last free variable inward and lands on an order
        # the result's own storage-order rules refuse, so the repair has a
        # pre-forced order to fall back to and these cells reach a rule that
        # describes their SHAPE.  Each carries a different code, which is the
        # point: the repair's generality is what makes three distinct
        # shape-specific diagnoses reachable instead of one order complaint.
        #
        # Eighteen of the original 28 are admitted since 2026-10-11: the four
        # ``unsupported_program_shape`` ones and the eight dense-driven
        # ``unsupported_sparse_output_domain`` ones (the ``WIDENED`` cells
        # above), and the six interleaved-key ``ijk->j`` ones, which
        # ``test_loopir_interleaved_workspace_key.py`` executes.  What remains
        # is the CSR-reduction residue.
        #
        # This is the whole lockable set at this tip, enumerated rather than
        # sampled: every rank-2 and rank-3 reduction cell for which the repair
        # precondition holds (forced order differs from pre-forced), whose code
        # is shape-specific, and which is arm-invariant.  §53.7's fourth code,
        # ``unsupported_union_with_dense``, needs a two-operand combiner and so
        # cannot be expressed through ``reduction_cin``; it stays uncovered here
        # and that is stated rather than left to look complete.
        (("dds", "ds", "ijk", "ij"), "unsupported_sparse_output_reduction"),
        (("dds", "sd", "ijk", "ij"), "unsupported_sparse_output_reduction"),
        (("dsd", "ds", "ijk", "ij"), "unsupported_sparse_output_reduction"),
        (("dsd", "sd", "ijk", "ij"), "unsupported_sparse_output_reduction"),
        (("dss", "ds", "ijk", "ij"), "unsupported_sparse_output_reduction"),
        (("dss", "sd", "ijk", "ij"), "unsupported_sparse_output_reduction"),
        (("sds", "ds", "ijk", "ij"), "unsupported_sparse_output_reduction"),
        (("sds", "sd", "ijk", "ij"), "unsupported_sparse_output_reduction"),
        (("ssd", "sd", "ijk", "ij"), "unsupported_sparse_output_reduction"),
        (("sss", "sd", "ijk", "ij"), "unsupported_sparse_output_reduction"),
    ],
    ids=lambda value: str(value),
)
def test_redispositioned_neighbours_keep_their_shape_specific_codes(cell, code):
    """§53.7's moved neighbours: the whole lockable set, not a sample.

    The repair is deliberately general -- it fires wherever the forced reorder
    produced an order the storage-order rules refuse and the unforced order is
    legal -- so 57 refused neighbours gained a code naming their own violation.
    That improvement was measured over the frontier and only two cells were
    locked.  All 28 that are expressible as one-operand reductions are locked
    here, spanning the three codes reachable that way, and the measured
    precondition for each is that ``select_loop_order``'s forced order really
    does differ from its pre-forced one; a cell where the two agree cannot be in
    the moved set at all.

    Blocker 3 and the 2026-10-11 dense-domain and interleaved-key changes have
    since ADMITTED part of §53.7's original 57, so this set is the
    lockable remainder at this tip rather than a subset of that historical
    number -- which is why each cell's code is asserted rather than assumed from
    the §53.7 split.
    """

    operand_fmt, result_fmt, operand_indices, result_indices = cell
    options = auto_options(False)
    pre_forced = []
    forced = Scheduler.select_loop_order(
        normalize_cin(
            reduction_cin(
                operand_fmt, result_fmt, operand_indices, result_indices, torch.float32
            ),
            compile_options=options,
        ),
        costs=options.scheduler.cost_model,
        pre_forced_order=pre_forced,
    )
    assert [v.name for v in forced] != [v.name for v in pre_forced]

    with pytest.raises((LoopIRLoweringError, LoopIRTargetError)) as error:
        compiled(cell, False)
    assert error.value.defect.code == code


def test_the_legacy_refusal_is_structured_and_not_repair_induced():
    """Legacy's own default-path limit, refused with a code at its own site.

    §53.8 records this refusal as a degradation the plan-replay boundary
    introduced: ``apply_schedule`` propagates only the plan, the legacy lowering
    replays it, and the repaired order has no legacy form.  That mechanism is
    not what happens.  The same refusal is raised from the same frame with
    **no requested schedule at all** -- that is, on plain default dispatch,
    which never consults ``apply_schedule`` and so cannot be replaying anything.

    The refusal belongs to the legacy CIN lowerer, not to the replay boundary:
    ``lower_TensorAssign`` reaches a compressed result's coordinate write with a
    REDUCED index variable innermost, which has no result level.  That used to
    escape as ``list.index``'s unstructured ``ValueError`` from
    ``level_of_index_var``; it is now refused there with the structured
    ``unscheduled_sparse_result_reduction`` code, so every legacy outcome on
    this surface is classified.

    The cell is chosen so the contrast is visible in one test: the TYPED route
    admits this program, so a cutover would replace a legacy refusal with a
    working kernel here, not merely with a different diagnostic.
    """

    import traceback

    from scorch.compiler.diagnostics import UnsupportedFeature
    from scorch.compiler.loopir.pipeline import legacy_generated_cpp

    cell = ("ss", "s", "ij", "i")
    operand_fmt, result_fmt, operand_indices, result_indices = cell
    shape, result_shape = shapes_for(operand_indices, result_indices)

    def legacy_frame(options):
        with pytest.raises(UnsupportedFeature) as error:
            legacy_generated_cpp(
                reduction_cin(
                    operand_fmt,
                    result_fmt,
                    operand_indices,
                    result_indices,
                    torch.float32,
                ),
                result_shape,
                ((shape, torch.float32),),
                compile_options=options,
            )
        assert [d.code for d in error.value.diagnostics] == [
            "unscheduled_sparse_result_reduction"
        ]
        return traceback.extract_tb(error.tb)[-1]

    default_dispatch = CompileOptions.from_environment(
        environ={}
    ).with_regblock_enabled(False)
    assert default_dispatch.requested_schedule is None

    without_schedule = legacy_frame(default_dispatch)
    with_empty_schedule = legacy_frame(auto_options(False))

    # Same raising frame with and without a schedule: the boundary is not the
    # cause.  The refusal is raised by the legacy lowerer itself.
    assert without_schedule.name == "lower_TensorAssign"
    assert with_empty_schedule.name == "lower_TensorAssign"
    assert without_schedule.filename.endswith("compiler/cin_lowerer.py")
    assert with_empty_schedule.filename == without_schedule.filename

    # And the typed route does not merely diagnose this shape better: it
    # compiles it.
    assert compiled(cell, False).cpp_source


# -- the gating ---------------------------------------------------------------


@pytest.mark.parametrize("cell", MIGRATED, ids=lambda c: f"{c[0]} {c[2]}->{c[3]}")
def test_the_two_halves_gate_together(cell):
    """Each half is necessary, stated as the two facts this process can prove.

    *The family alone is not sufficient*: the order the forced reorder produces
    is refused by the plan's own legality rules, so without the repair the
    automatic arm never reaches the family at all -- it stops at
    ``sparse_parent_dominance``, which is asserted here directly against the
    legality boundary rather than inferred.

    *The repair alone is not sufficient*: the order it restores is the cell's
    declared order, and that order only compiles because the family exists --
    which the declared-order compile below exercises.  (That the repair alone
    migrates nothing is measured out of process, against a source tree with the
    family reverted; it cannot be stated in one interpreter.)
    """

    from scorch.compiler.loop_plan import LoopPlan, verify_loop_plan

    operand_fmt, result_fmt, operand_indices, result_indices = cell
    options = auto_options(False)
    normalized = normalize_cin(
        reduction_cin(
            operand_fmt, result_fmt, operand_indices, result_indices, torch.float32
        ),
        compile_options=options,
    )
    costs = options.scheduler.cost_model
    pre_forced = []
    forced = Scheduler.select_loop_order(
        normalized, costs=costs, pre_forced_order=pre_forced
    )

    def plan_for(order):
        return LoopPlan(
            loop_order=tuple(v.index_id for v in order),
            auto_policy=Scheduler._auto_origin_policy(options.scheduler),
            provenance="auto",
        )

    with pytest.raises(InvalidSchedule) as error:
        verify_loop_plan(normalized, plan_for(forced))
    assert [d.code for d in error.value.diagnostics] == ["sparse_parent_dominance"]

    repaired = verify_loop_plan(normalized, plan_for(pre_forced))
    assert repaired.workspace is None
    assert repaired.tiles == ()

    assert compiled(cell, False) is not None
    assert compiled(cell, False, options=declared_options(operand_indices)) is not None


@pytest.mark.parametrize("cell", MIGRATED, ids=lambda c: f"{c[0]} {c[2]}->{c[3]}")
def test_the_repair_restores_the_pre_forced_order(cell):
    """The repaired plan is exactly the composition's own pre-forced order."""

    operand_fmt, result_fmt, operand_indices, result_indices = cell
    dtype = torch.float32
    shape, result_shape = shapes_for(operand_indices, result_indices)
    # ONE CIN for both measurements: stable IDs are preserved by normalization's
    # clone, but a second construction allocates fresh ones.
    cin = reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices, dtype)
    options = auto_options(False)
    costs = options.scheduler.cost_model
    pre_forced = []
    forced = Scheduler.select_loop_order(
        normalize_cin(cin, compile_options=options),
        costs=costs,
        pre_forced_order=pre_forced,
    )
    assert [v.name for v in pre_forced] != [v.name for v in forced]
    # The pre-forced order is the CIN's own declared nest order here.
    assert [v.name for v in pre_forced] == list(operand_indices)

    kernel = compile_cin_via_loopir(
        cin, result_shape, ((shape, dtype),), compile_options=options
    )
    recorded = kernel.schedule.plan.loop_order
    assert list(recorded) == [v.index_id for v in pre_forced]
    assert list(recorded) != [v.index_id for v in forced]


def test_the_out_parameter_defaults_to_leaving_the_legacy_path_alone():
    """``select_loop_order`` without the sink returns the forced order."""

    options = auto_options(False)
    normalized = normalize_cin(
        reduction_cin("ss", "s", "ij", "i", torch.float32),
        compile_options=options,
    )
    costs = options.scheduler.cost_model
    plain = Scheduler.select_loop_order(normalized, costs=costs)
    sink = []
    withsink = Scheduler.select_loop_order(
        normalized, costs=costs, pre_forced_order=sink
    )
    assert [v.name for v in plain] == [v.name for v in withsink]
    assert [v.name for v in sink] != [v.name for v in plain]


@pytest.mark.parametrize(
    "bad", [(), "ij", ["i"], None], ids=["tuple", "str", "nonempty", "none-ish"]
)
def test_the_out_parameter_is_validated(bad):
    if bad is None:
        pytest.skip("None means 'no sink', which is the default path")
    options = auto_options(False)
    normalized = normalize_cin(
        reduction_cin("ss", "s", "ij", "i", torch.float32),
        compile_options=options,
    )
    with pytest.raises(TypeError):
        Scheduler.select_loop_order(
            normalized, costs=options.scheduler.cost_model, pre_forced_order=bad
        )


def test_the_repair_only_answers_an_order_legality_refusal():
    """A refusal about anything other than the order is re-raised untouched."""

    class Fake:
        def __init__(self, code):
            self.code = code

    error = InvalidSchedule("x", diagnostics=(Fake("auto_tile_decision"),))
    assert not Scheduler._refused_for_order_legality(error)
    for code in ("result_storage_order", "sparse_parent_dominance"):
        assert Scheduler._refused_for_order_legality(
            InvalidSchedule("x", diagnostics=(Fake(code),))
        )
    # A refusal carrying no diagnostics at all is not an order refusal.
    assert not Scheduler._refused_for_order_legality(InvalidSchedule("x"))


def test_the_repaired_plan_records_no_workspace_and_no_tiles():
    """The legality layer re-derives both facts from the repaired order."""

    kernel = compiled(("sss", "s", "ijk", "i"), False)
    plan = kernel.schedule.plan if kernel.schedule is not None else None
    if plan is None:  # pragma: no cover - the scheduled path always records one
        pytest.skip("no plan recorded on this route")
    assert plan.workspace is None
    assert plan.tiles == ()
    assert plan.provenance == "auto"
