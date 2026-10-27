"""Each rule on which loops may, or must, be marked parallel has one code and one
exception type on every route.

The rules: a mark is refused on a reduction loop (``parallel_reduction``), on the ragged
inner loop of an affine tile (``parallel_ragged_inner``), anywhere when the result is
sparse (``parallel_sparse_result``), on a sparse panel tile (``parallel_panel_loop``) and
on a heap tile's loops (``heap_parallel_tile``); a sparse panel tile needs its dense
parent row as the parallel loop (``panel_parallel_parent``), heap accumulation needs a
parallel loop (``heap_parallel_prefix``) on a dense result-prefix axis
(``heap_parallel_ownership``), and a packed relayout needs its CSR row as the parallel
loop (``relayout_parallel_row``).  The Schedule route checks them on the schedule's loop
names before a plan exists (``Scheduler._apply_schedule_legacy``), the plan route checks
them on a ``LoopPlan`` (``loop_plan_legality``), and the direct CIN route's lowering
checks the reduction rule on a hand-marked ``ForAll``.

Before, the Schedule route raised ``ScorchValueError`` or ``ScorchNotImplementedError``
with codes of its own (``parallel_reduction_loop_invalid``,
``parallel_ragged_tile_invalid``, ``explicit_parallel_sparse_result_unsupported``,
``panel_parallel_loop_invalid`` for both panel rules, ``heap_tile_schedule_invalid`` for
the heap rules and ``relayout_schedule_invalid``), and the plan route's errors carried
their code only in their diagnostics.  Now every route raises through
``loop_plan_legality.refuse_parallel_mark``: the
rule's type, ``code`` set, and diagnostics that each name the code.  No refused program
reaches a kernel build, in either automatic arm.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Tuple

import pytest
import torch

from scorch import STensor, ops  # type: ignore[import-untyped]
from scorch.compiler.cin import (  # type: ignore[import-untyped]
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.cin_analysis import analyze_cin  # type: ignore[import-untyped]
from scorch.compiler.cin_lowerer import CINLowerer  # type: ignore[import-untyped]
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    InvalidSchedule,
    UnsupportedFeature,
    VerificationError,
)
from scorch.compiler.loop_plan import (  # type: ignore[import-untyped]
    LoopPart,
    LoopRef,
    ScheduledCIN,
    verify_loop_plan,
    verify_scheduled_cin,
)
from scorch.compiler.loop_plan_legality import (  # type: ignore[import-untyped]
    HEAP_PARALLEL_OWNERSHIP,
    HEAP_PARALLEL_PREFIX,
    HEAP_PARALLEL_TILE,
    PANEL_PARALLEL_PARENT,
    PARALLEL_PANEL_LOOP,
    PARALLEL_RAGGED_INNER,
    PARALLEL_REDUCTION,
    PARALLEL_SPARSE_RESULT,
    LoopPlanDiagnostic,
    refuse_parallel_mark,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    RelayoutSpec,
    Schedule,
    Scheduler,
    TileSpec,
)
from scorch.exceptions import (  # type: ignore[import-untyped]
    ScorchAssertionError,
    defect_code,
)

F64 = torch.float64
RULE_TYPES = {
    PARALLEL_REDUCTION: InvalidSchedule,
    PARALLEL_RAGGED_INNER: InvalidSchedule,
    PARALLEL_SPARSE_RESULT: UnsupportedFeature,
    PARALLEL_PANEL_LOOP: InvalidSchedule,
    PANEL_PARALLEL_PARENT: InvalidSchedule,
    HEAP_PARALLEL_TILE: InvalidSchedule,
    HEAP_PARALLEL_PREFIX: InvalidSchedule,
    HEAP_PARALLEL_OWNERSHIP: InvalidSchedule,
}
CASE_IDS = [
    "reduction loop",
    "ragged inner loop of a tile",
    "sparse result",
    "panel tile",
    "panel without its parent row",
    "heap tile marked parallel",
    "parallel loop on the heap tile",
    "heap tile without a parallel loop",
]


def _set_arm(monkeypatch: pytest.MonkeyPatch, arm: int) -> None:
    monkeypatch.setenv("SCORCH_REGBLOCK", str(arm))
    assert CompileOptions.from_environment().scheduler.regblock_enabled == bool(arm)


def _forbid_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a refused program reached the kernel build")

    monkeypatch.setattr(ops, "_prepare_generated_kernel_build", forbidden)


def _spmm(result_format: str = "dd") -> ForAll:
    """``C(i, k) += A(i, j) * B(j, k)``, A CSR (``ds``), B dense; loop order i, j, k.
    ``j`` is the reduction variable."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt=result_format)
    a = TensorVar("A", fmt="ds")
    b = TensorVar("B", fmt="ds" if result_format == "ds" else "dd")
    c[i, k] = a[i, j] * b[j, k]
    return ForAll(i, ForAll(j, ForAll(k, c._assignment)))


def _index_ids(cin: Any) -> Dict[str, Any]:
    analysis = analyze_cin(cin)
    return {
        definition.display_name: index_id
        for index_id, definition in analysis.index_definitions.items()
        if definition.bindings
    }


def _assert_refusal(error: BaseException, code: str) -> None:
    assert type(error) is RULE_TYPES[code]
    assert getattr(error, "code", None) == code
    assert defect_code(error) == code
    diagnostics = getattr(error, "diagnostics", ())
    assert diagnostics
    assert [diagnostic.code for diagnostic in diagnostics] == [code] * len(diagnostics)


# -- the Schedule route: a Schedule's mark, refused before a plan exists --------------

PANEL = TileSpec("j", 32, kind="panel", accum="direct", parallel=True)
# A heap-accumulated tile of the result's trailing axis k; with parallel_loop="i" the
# Schedule route accepts it.
HEAP = TileSpec("k", 4, placement="outermost", accum="heap", unroll=False)
SCHEDULE_CASES: Dict[
    str, Tuple[str, Callable[[], ForAll], Schedule, Tuple[str, ...]]
] = {
    # id: (code, program, schedule, path of each mark's diagnostic)
    "reduction loop": (
        PARALLEL_REDUCTION,
        _spmm,
        Schedule(parallel_loop="j"),
        ("schedule", "parallel_loop"),
    ),
    "ragged inner loop of a tile": (
        PARALLEL_RAGGED_INNER,
        _spmm,
        Schedule(tiles=(TileSpec("k", 4, accum="direct"),), parallel_loop="k_in"),
        ("schedule", "parallel_loop"),
    ),
    "sparse result": (
        PARALLEL_SPARSE_RESULT,
        lambda: _spmm("ds"),
        Schedule(parallel_loop="i"),
        ("schedule", "parallel_loop"),
    ),
    "panel tile": (
        PARALLEL_PANEL_LOOP,
        _spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(PANEL,)),
        ("schedule", "tiles", "0", "parallel"),
    ),
    "panel without its parent row": (
        PANEL_PARALLEL_PARENT,
        _spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(replace(PANEL, parallel=False),)),
        ("schedule", "parallel_loop"),
    ),
    "heap tile marked parallel": (
        HEAP_PARALLEL_TILE,
        _spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(replace(HEAP, parallel=True),)),
        ("schedule", "tiles", "0", "parallel"),
    ),
    "parallel loop on the heap tile": (
        HEAP_PARALLEL_TILE,
        _spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(HEAP,), parallel_loop="k_out"),
        ("schedule", "parallel_loop"),
    ),
    "heap tile without a parallel loop": (
        HEAP_PARALLEL_PREFIX,
        _spmm,
        Schedule(loop_order=("i", "j", "k"), tiles=(HEAP,)),
        ("schedule", "parallel_loop"),
    ),
}


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("case", list(SCHEDULE_CASES))
def test_the_schedule_route_refuses_by_the_rule(
    monkeypatch: pytest.MonkeyPatch, arm: int, case: str
) -> None:
    _set_arm(monkeypatch, arm)
    _forbid_builds(monkeypatch)
    code, program, schedule, path = SCHEDULE_CASES[case]
    with pytest.raises(RULE_TYPES[code]) as caught:
        Scheduler.apply_schedule(program(), schedule)
    _assert_refusal(caught.value, code)
    assert [diagnostic.path for diagnostic in caught.value.diagnostics] == [path]
    assert caught.value.diagnostics[0].stage == "schedule"
    assert str(caught.value).startswith(f"stage=Schedule: {code} at {'/'.join(path)}: ")


def test_the_schedule_route_keeps_its_messages() -> None:
    """The text after the code is the text the Schedule route raised before."""

    expected = {
        "reduction loop": "Reduction loops cannot be selected for parallel execution: "
        "['j']",
        "ragged inner loop of a tile": "Tiled inner loops contain a ragged-tail break "
        "and cannot be parallelized: ['k_in']",
        "sparse result": "Explicit parallel-loop selection currently requires a dense "
        "result tensor",
        "panel tile": "Sparse panel outer loops must be serial; select the row loop "
        "with Schedule.parallel_loop",
        "panel without its parent row": "Sparse panel tiling requires its CSR "
        "dense-parent row loop 'i' as Schedule.parallel_loop",
        "heap tile marked parallel": "A heap-backed result tile uses shared reusable "
        "storage; its outer tile loop must be serial",
        "parallel loop on the heap tile": "A heap-backed result tile cannot select its "
        "shared tile loop for parallel execution",
        "heap tile without a parallel loop": "Heap accumulation requires an explicit "
        "parallel dense result-prefix loop so the shared outer result-tile loop "
        "remains serial",
    }
    for case, text in expected.items():
        _, program, schedule, _ = SCHEDULE_CASES[case]
        with pytest.raises((InvalidSchedule, UnsupportedFeature)) as caught:
            Scheduler.apply_schedule(program(), schedule)
        assert str(caught.value).endswith(text)
        assert caught.value.diagnostics[0].message == text


@pytest.mark.parametrize("arm", [0, 1])
def test_a_public_matmul_gets_the_same_refusal(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    _forbid_builds(monkeypatch)
    generator = torch.Generator().manual_seed(18)
    a = torch.rand(7, 6, generator=generator, dtype=F64)
    a[a < 0.5] = 0.0
    b = torch.rand(6, 5, generator=generator, dtype=F64)
    # matmul names its product "ij,jk->ik", so j is the reduction.
    schedule = Schedule(loop_order=("i", "j", "k"), parallel_loop="j", tag="par-j")
    with pytest.raises(InvalidSchedule) as caught:
        ops.matmul(a.to_sparse_csr(), b, use_cache=False, schedule=schedule)
    _assert_refusal(caught.value, PARALLEL_REDUCTION)


# -- the plan route: a LoopPlan whose mark breaks the rule ----------------------------


def _plan_cases() -> Dict[str, Tuple[str, Any, Any]]:
    """id: (code, normalized CIN, forged plan), each from a plan the Schedule route
    verified, with one field changed."""

    cases: Dict[str, Tuple[str, Any, Any]] = {}

    scheduled = Scheduler.apply_schedule(_spmm(), Schedule(loop_order=("i", "j", "k")))
    ids = _index_ids(scheduled.normalized_cin)
    cases["reduction loop"] = (
        PARALLEL_REDUCTION,
        scheduled.normalized_cin,
        replace(scheduled.verified_loop_plan, parallel_loop=LoopRef(ids["j"])),
    )

    scheduled = Scheduler.apply_schedule(
        _spmm(),
        Schedule(loop_order=("i", "j", "k"), tiles=(TileSpec("k", 4, accum="direct"),)),
    )
    ids = _index_ids(scheduled.normalized_cin)
    cases["ragged inner loop of a tile"] = (
        PARALLEL_RAGGED_INNER,
        scheduled.normalized_cin,
        replace(
            scheduled.verified_loop_plan,
            parallel_loop=LoopRef(ids["k"], LoopPart.INNER),
        ),
    )

    scheduled = Scheduler.apply_schedule(_spmm("ds"), Schedule())
    ids = _index_ids(scheduled.normalized_cin)
    cases["sparse result"] = (
        PARALLEL_SPARSE_RESULT,
        scheduled.normalized_cin,
        replace(scheduled.verified_loop_plan, parallel_loop=LoopRef(ids["i"])),
    )

    scheduled = Scheduler.apply_schedule(
        _spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(replace(PANEL, parallel=False),),
            parallel_loop="i",
        ),
    )
    plan = scheduled.verified_loop_plan
    cases["panel tile"] = (
        PARALLEL_PANEL_LOOP,
        scheduled.normalized_cin,
        replace(
            plan,
            parallel_loop=None,
            tiles=tuple(replace(tile, parallel=True) for tile in plan.tiles),
        ),
    )
    ids = _index_ids(scheduled.normalized_cin)
    cases["panel without its parent row"] = (
        PANEL_PARALLEL_PARENT,
        scheduled.normalized_cin,
        replace(plan, parallel_loop=LoopRef(ids["k"])),
    )

    scheduled = Scheduler.apply_schedule(
        _spmm(), Schedule(loop_order=("i", "j", "k"), tiles=(HEAP,), parallel_loop="i")
    )
    plan = scheduled.verified_loop_plan
    ids = _index_ids(scheduled.normalized_cin)
    cases["heap tile marked parallel"] = (
        HEAP_PARALLEL_TILE,
        scheduled.normalized_cin,
        replace(
            plan,
            parallel_loop=None,
            tiles=tuple(replace(tile, parallel=True) for tile in plan.tiles),
        ),
    )
    cases["parallel loop on the heap tile"] = (
        HEAP_PARALLEL_TILE,
        scheduled.normalized_cin,
        replace(plan, parallel_loop=LoopRef(ids["k"])),
    )
    cases["heap tile without a parallel loop"] = (
        HEAP_PARALLEL_PREFIX,
        scheduled.normalized_cin,
        replace(plan, parallel_loop=None),
    )
    return cases


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("case", CASE_IDS)
def test_the_plan_route_refuses_by_the_rule(
    monkeypatch: pytest.MonkeyPatch, arm: int, case: str
) -> None:
    _set_arm(monkeypatch, arm)
    _forbid_builds(monkeypatch)
    code, cin, plan = _plan_cases()[case]
    for check in (
        lambda: verify_loop_plan(cin, plan),
        lambda: verify_scheduled_cin(ScheduledCIN(cin, plan)),
    ):
        with pytest.raises(RULE_TYPES[code]) as caught:
            check()
        _assert_refusal(caught.value, code)
        assert str(caught.value).startswith(f"stage=LoopPlan: {code} at ")
        assert caught.value.diagnostics[0].stage == "loop_plan"
        if case in ("panel tile", "heap tile marked parallel"):
            assert caught.value.diagnostics[0].path == ("tiles", "0", "parallel")
        # A missing parallel loop names no loop.
        if code != HEAP_PARALLEL_PREFIX:
            assert caught.value.diagnostics[0].index_id is not None


# -- rules an earlier check shadows on these programs ---------------------------------


def test_a_plan_that_marks_a_tile_and_names_a_parallel_loop_is_malformed() -> None:
    """``LoopPlan`` selects parallelism on the plan or on a tile, not both, and its
    structural check refuses a plan that does both before any legality rule.  So the
    panel check's own refusal of a parallel panel tile is reached by no plan: with no
    ``parallel_loop`` the tile is the plan's parallel loop, and
    ``_effective_parallel_loop`` refuses it first (``parallel_panel_loop``)."""

    scheduled = Scheduler.apply_schedule(
        _spmm(),
        Schedule(
            loop_order=("i", "j", "k"),
            tiles=(replace(PANEL, parallel=False),),
            parallel_loop="i",
        ),
    )
    plan = scheduled.verified_loop_plan
    both = replace(
        plan, tiles=tuple(replace(tile, parallel=True) for tile in plan.tiles)
    )
    with pytest.raises(VerificationError) as caught:
        verify_loop_plan(scheduled.normalized_cin, both)
    assert defect_code(caught.value) == "loop_plan_reference_invalid"


def test_a_relayout_without_its_row_loop_is_refused_by_the_panel_rule() -> None:
    """A packed relayout needs exactly one panel tile, on the CSR operand, whose dense
    parent is the relayout's CSR row.  So the panel's rule refuses a relayout without
    its row loop on both routes before ``relayout_parallel_row`` is checked."""

    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec("k", 4, placement="outermost", accum="direct", unroll=False),
            TileSpec("j", 3, placement="child_of:k_out", kind="panel", accum="direct"),
        ),
        relayout=RelayoutSpec("B", "k", 4, scope_var="j"),
        parallel_loop="i",
    )
    scheduled = Scheduler.apply_schedule(_spmm(), schedule)
    with pytest.raises(InvalidSchedule) as refused:
        Scheduler.apply_schedule(_spmm(), replace(schedule, parallel_loop=None))
    _assert_refusal(refused.value, PANEL_PARALLEL_PARENT)
    with pytest.raises(InvalidSchedule) as forged:
        verify_loop_plan(
            scheduled.normalized_cin,
            replace(scheduled.verified_loop_plan, parallel_loop=None),
        )
    _assert_refusal(forged.value, PANEL_PARALLEL_PARENT)


def test_a_heap_mark_on_a_name_no_loop_has_is_refused_by_ownership() -> None:
    """Every free axis but the heap tile's is a result-prefix axis, and the reduction
    and heap-tile rules come first, so on the Schedule route ``heap_parallel_ownership``
    refuses only a parallel loop that names no loop: the heap checks run before the
    check that the name exists."""

    with pytest.raises(InvalidSchedule) as caught:
        Scheduler.apply_schedule(
            _spmm(),
            Schedule(loop_order=("i", "j", "k"), tiles=(HEAP,), parallel_loop="r"),
        )
    _assert_refusal(caught.value, HEAP_PARALLEL_OWNERSHIP)
    assert caught.value.diagnostics[0].path == ("schedule", "parallel_loop")


# -- the direct route: a hand-marked reduction loop ------------------------------------


def _hand_marked_reduction() -> ForAll:
    """``C(i, j) += A(i, k) * B(k, j)``, all dense, with ``k`` outermost and marked."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dd", shape=(7, 5), dtype=F64)
    a = TensorVar("A", fmt="dd", shape=(7, 6), dtype=F64)
    b = TensorVar("B", fmt="dd", shape=(6, 5), dtype=F64)
    body = ForAll(j, TensorAssign(c[i, j], a[i, k] * b[k, j], op=Operation.ADD))
    return ForAll(k, ForAll(i, body), parallel=True)


@pytest.mark.parametrize("arm", [0, 1])
def test_the_direct_route_refuses_by_the_rule(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    _forbid_builds(monkeypatch)
    generator = torch.Generator().manual_seed(18)
    a = STensor.from_torch(torch.rand(7, 6, generator=generator, dtype=F64))
    b = STensor.from_torch(torch.rand(6, 5, generator=generator, dtype=F64))
    with pytest.raises(InvalidSchedule) as caught:
        ops.lower_and_exec_cin(_hand_marked_reduction(), (7, 5), a, b)
    _assert_refusal(caught.value, PARALLEL_REDUCTION)
    assert caught.value.diagnostics[0].stage == "legacy_cin_lowering"
    with pytest.raises(InvalidSchedule) as lowered:
        CINLowerer().lower_IndexStmt(_hand_marked_reduction())
    _assert_refusal(lowered.value, PARALLEL_REDUCTION)


# -- the shared raise refuses its own misuse --------------------------------------------


def test_refuse_parallel_mark_refuses_an_unknown_rule() -> None:
    diagnostic = LoopPlanDiagnostic(code="parallel_somewhere", message="m", path=())
    with pytest.raises(ScorchAssertionError) as caught:
        refuse_parallel_mark("parallel_somewhere", "m", (diagnostic,))
    assert defect_code(caught.value) == "parallel_mark_rule_unknown"


@pytest.mark.parametrize("diagnostics", ["none", "another code"])
def test_refuse_parallel_mark_needs_diagnostics_that_name_the_rule(
    diagnostics: str,
) -> None:
    given: Tuple[object, ...] = (
        ()
        if diagnostics == "none"
        else (LoopPlanDiagnostic(code=PARALLEL_RAGGED_INNER, message="m", path=()),)
    )
    with pytest.raises(ScorchAssertionError) as caught:
        refuse_parallel_mark(PARALLEL_REDUCTION, "m", given)
    assert defect_code(caught.value) == "parallel_mark_diagnostics_invalid"
