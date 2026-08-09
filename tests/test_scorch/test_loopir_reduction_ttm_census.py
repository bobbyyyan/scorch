"""Phase-7 cluster 2: the multi-compressed reduction / TTM fail-closed census.

Cluster 2 -- sparse reductions and TTM over multi-compressed receivers -- is
**not migrated**.  This module pins the exact boundary each cell stops at, in
both automatic arms, so the next migration slice starts from a locked map
rather than a re-derived one, and so no cell can silently change its
diagnosis.

The cells split into two groups, and the split is the point.

**Reorder-blocked (5 cells).**  ``Scheduler.select_loop_order`` ends with a
forced reorder (``scheduler.py``, "ensure at least one free variable appears
after the last reduction variable"): when no free variable follows the last
reduction it moves the last free variable to the very end of the loop order,
with no legality check.  For these cells that permutation violates the sparse
operand's own parent dominance -- ``sss ijk->ij`` declared ``i,j,k`` becomes
``i,k,j`` while ``A``'s storage order is ``i,j,k`` -- so the LoopIR route
rejects the plan at ``loop_plan_legality``'s ``sparse_parent_dominance``
before any LoopIR admission decision is reached.

That reorder exists precisely because the legacy ``insert_workspace`` can only
key a workspace on free variables *below the innermost reduction*.  It is a
fifth blocker beside the four recorded in review section 45.6, and it is not
LoopIR's to fix: ``Scheduler.apply_schedule`` is shared with legacy dispatch,
legacy produces numerically correct results from the reordered order today,
and changing it would change generated code on the default path.

**Reachable (7 cells).**  Their automatic plan order is legal, and they stop
at LoopIR-side admission walls instead.  These are the cells a migration slice
can actually take, and between them they span every shape the vertical needs:
rank-1 and rank-2 keys, with and without a bound prefix, single-cursor and
merged producers.

Applying the anchoring rule -- anchor the region at the OUTERMOST reduction
and key it on the result indices at or below that anchor, in result level
order -- to the reachable cells gives:

===================  ==========  =====  ========  ======  ===
cell                 reductions  p      prefix    key     K
===================  ==========  =====  ========  ======  ===
``sss ijk->k``       i, j        0      --        k       1
``sss ijk->ik``      j           1      i         k       1
``sss ijk->jk``      i           0      --        j, k    2
``ss ij->j``         i           0      --        j       1
``ds ij->j``         i           0      --        j       1
``TTM ijk,kl->ijl``  k           2      i, j      l       1
===================  ==========  =====  ========  ======  ===

B1 SpGEMM (``ik,kj->ij``) is the same rule's ``K = 1, prefix = 1`` instance,
which is why the migrated family is exactly the K == 1 case of the ordered
key domain rather than a separate form.
"""

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
from scorch.compiler.loop_plan_legality import InvalidSchedule
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir
from scorch.compiler.loopir.schedule_passes import SchedulePassError
from tests.test_scorch.test_loopir_sparse_workspace_target import auto_options


def reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices):
    """``C[result] += A[operand]`` over one shared index nest."""

    order = operand_indices
    ivars = {name: IndexVar(name) for name in order}
    operand = TensorVar("A", fmt=operand_fmt, dtype=torch.float32)[
        tuple(ivars[name] for name in operand_indices)
    ]
    result = TensorVar("C", fmt=result_fmt, dtype=torch.float32)[
        tuple(ivars[name] for name in result_indices)
    ]
    stmt = TensorAssign(result, operand, op=Operation.ADD)
    for name in reversed(order):
        stmt = ForAll(ivars[name], stmt)
    return stmt


def ttm_cin(a_fmt, b_fmt, c_fmt):
    """``C[i,j,l] += A[i,j,k] * B[k,l]`` -- tensor times matrix."""

    ivars = {name: IndexVar(name) for name in "ijkl"}
    a = TensorVar("A", fmt=a_fmt, dtype=torch.float32)[
        ivars["i"], ivars["j"], ivars["k"]
    ]
    b = TensorVar("B", fmt=b_fmt, dtype=torch.float32)[ivars["k"], ivars["l"]]
    c = TensorVar("C", fmt=c_fmt, dtype=torch.float32)[
        ivars["i"], ivars["j"], ivars["l"]
    ]
    stmt = TensorAssign(c, CINBinaryOp(Operation.MUL, a, b), op=Operation.ADD)
    for name in reversed("ijkl"):
        stmt = ForAll(ivars[name], stmt)
    return stmt


_SHAPE_3 = (4, 5, 6)
_SHAPE_2 = (4, 5)
_F32 = torch.float32


def _reduction_cell(name, operand_fmt, result_fmt, operand_indices, result_indices):
    shape = _SHAPE_3 if len(operand_indices) == 3 else _SHAPE_2
    result_shape = tuple(shape[operand_indices.index(c)] for c in result_indices)
    return (
        name,
        reduction_cin(operand_fmt, result_fmt, operand_indices, result_indices),
        result_shape,
        ((shape, _F32),),
    )


def _ttm_cell(name, a_fmt, b_fmt, c_fmt):
    return (
        name,
        ttm_cin(a_fmt, b_fmt, c_fmt),
        (4, 5, 3),
        ((_SHAPE_3, _F32), ((6, 3), _F32)),
    )


# The plan order these cells receive violates the operand's parent dominance,
# so they never reach a LoopIR admission decision.  ``InvalidSchedule`` carries
# no ``defect.code``; the stage and reason are checked from its message.
REORDER_BLOCKED = [
    _reduction_cell("sss ijk->i", "sss", "s", "ijk", "i"),
    _reduction_cell("sss ijk->j", "sss", "s", "ijk", "j"),
    _reduction_cell("sss ijk->ij", "sss", "ss", "ijk", "ij"),
    _reduction_cell("ss ij->i", "ss", "s", "ij", "i"),
    _reduction_cell("ds ij->i", "ds", "s", "ij", "i"),
]

# Legal plan order; stopped by a LoopIR-side admission wall with an exact code.
REACHABLE = [
    (
        *_reduction_cell("sss ijk->k", "sss", "s", "ijk", "k"),
        "unsupported_sparse_output",
    ),
    (
        *_reduction_cell("sss ijk->ik", "sss", "ss", "ijk", "ik"),
        "sparse_workspace_target_invalid",
    ),
    (
        *_reduction_cell("sss ijk->jk", "sss", "ss", "ijk", "jk"),
        "unsupported_schedule_auto_family",
    ),
    (*_reduction_cell("ss ij->j", "ss", "s", "ij", "j"), "unsupported_sparse_output"),
    (*_reduction_cell("ds ij->j", "ds", "s", "ij", "j"), "unsupported_sparse_output"),
    (
        *_ttm_cell("TTM sss x ss -> sss", "sss", "ss", "sss"),
        "unsupported_sparse_output",
    ),
    (
        *_ttm_cell("TTM dds x dd -> dds", "dds", "dd", "dds"),
        "unsupported_sparse_output",
    ),
]


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize(
    "cell", REORDER_BLOCKED, ids=[cell[0] for cell in REORDER_BLOCKED]
)
def test_reorder_blocked_cells_stop_at_parent_dominance(cell, arm):
    """The shared scheduler's forced reorder makes the plan itself illegal.

    These stop before LoopIR gets a say, which is why no LoopIR-side
    admission widening can reach them.
    """

    name, cin, result_shape, bindings = cell
    with pytest.raises(InvalidSchedule) as error:
        compile_cin_via_loopir(
            cin, result_shape, bindings, compile_options=auto_options(arm)
        )
    message = str(error.value)
    assert "stage=LoopPlan" in message, name
    assert "sparse_parent_dominance" in message, name


@pytest.mark.parametrize("arm", [False, True])
@pytest.mark.parametrize("cell", REACHABLE, ids=[cell[0] for cell in REACHABLE])
def test_reachable_cells_keep_their_exact_admission_code(cell, arm):
    """A legal plan order, stopped by a LoopIR admission wall with an exact code.

    Each of these is a cell a migration slice can take.  If one starts
    compiling, this lock must move to the neighbour that still occupies the
    seam rather than be deleted.
    """

    name, cin, result_shape, bindings, expected_code = cell
    with pytest.raises((LoopIRLoweringError, SchedulePassError)) as error:
        compile_cin_via_loopir(
            cin, result_shape, bindings, compile_options=auto_options(arm)
        )
    assert error.value.defect.code == expected_code, (name, error.value.defect)


def test_the_census_covers_the_whole_declared_cluster():
    """No cell may quietly leave the census."""

    assert len(REORDER_BLOCKED) == 5
    assert len(REACHABLE) == 7
    names = [cell[0] for cell in REORDER_BLOCKED] + [cell[0] for cell in REACHABLE]
    assert len(names) == len(set(names)) == 12
