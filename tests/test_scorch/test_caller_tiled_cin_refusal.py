"""A statement its caller tiled is refused at normalization, before any lowering.

``Scheduler.add_tile`` strip-mines a loop of a CIN statement in place: ``i`` becomes
``i_out + i_in``, a ``TileSizeVar`` gives both loops the tile's width, and the two loops
carry their split roles.  Normalized CIN carries no schedule, and ``normalize_cin``'s
clone kept the sum but cleared every width and role.  So on every entry that normalizes
(``ops.lower_and_exec_cin``, ``Scheduler.apply_schedule``, ``Scheduler.auto_schedule``)
the tile was lost: ``add_tile(cin, i, 4, parallel=True, use_workspace=False)`` on dense
``C(i, j) += A(i, k) * B(k, j)`` reached the C++ compiler as two loops over the whole
extent, with ``int pC0 = i;`` and ``i`` never declared, whatever ``parallel`` and
``use_workspace`` said and for a tile of ``k`` as well.  Now normalization refuses an
index variable that carries a sum, which only tiling gives it, with
``caller_tiled_cin_unsupported`` (an ``UnsupportedFeature``), one diagnostic per summed
index.  The fields it clears are still not read (``test_cin_analysis``'s
``test_normalization_discards_legacy_only_schedule_metadata``).  The same tile is requested
as ``Schedule(tiles=(TileSpec(...),))``, which the scheduler applies after
normalization.  ``CINLowerer.lower_IndexStmt`` keeps its own copy of the caller's tree
and still lowers the tile.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest
import torch

from scorch import STensor, ops  # type: ignore[import-untyped]
from scorch.compiler.cin import (  # type: ignore[import-untyped]
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    TileSizeVar,
    Where,
    Workspace,
)
from scorch.compiler.cin_analysis import normalize_cin  # type: ignore[import-untyped]
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    UnsupportedFeature,
    VerificationError,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    Schedule,
    Scheduler,
)
from scorch.exceptions import defect_code  # type: ignore[import-untyped]
from tests.test_scorch.test_helpers import lower_and_print

REFUSAL = "caller_tiled_cin_unsupported"
F64 = torch.float64
N, K, M = 7, 6, 5


def _set_arm(monkeypatch: pytest.MonkeyPatch, arm: int) -> None:
    monkeypatch.setenv("SCORCH_REGBLOCK", str(arm))
    assert CompileOptions.from_environment().scheduler.regblock_enabled == bool(arm)


def _forbid_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a refused program reached the kernel build")

    monkeypatch.setattr(ops, "_prepare_generated_kernel_build", forbidden)


def _program(layout: str) -> Tuple[ForAll, Dict[str, IndexVar]]:
    """Dense ``C(i, j) += A(i, k) * B(k, j)``, loop order i, k, j: with no workspace
    ("no_ws") or with a dense workspace over j ("dense_ws")."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dd", shape=(N, M), dtype=F64)
    a = TensorVar("A", fmt="dd", shape=(N, K), dtype=F64)
    b = TensorVar("B", fmt="dd", shape=(K, M), dtype=F64)
    if layout == "no_ws":
        body: Any = ForAll(
            j, TensorAssign(c[i, j], a[i, k] * b[k, j], op=Operation.ADD)
        )
    else:
        w = Workspace(name="w", dim=1, dtype=F64, dense=True)
        body = Where(
            producer=ForAll(j, TensorAssign(w[j], a[i, k] * b[k, j], op=Operation.ADD)),
            consumer=ForAll(j, TensorAssign(c[i, j], w[j], op=Operation.ADD)),
        )
    return ForAll(i, ForAll(k, body)), {"i": i, "j": j, "k": k}


def _tiled(layout: str, target: str, parallel: bool, use_workspace: bool) -> ForAll:
    cin, names = _program(layout)
    return Scheduler.add_tile(
        cin,
        names[target],
        4,
        placement="outermost",
        parallel=parallel,
        use_workspace=use_workspace,
    )


def _operands() -> Tuple[STensor, STensor]:
    generator = torch.Generator().manual_seed(22)
    a = torch.rand(N, K, generator=generator, dtype=F64)
    b = torch.rand(K, M, generator=generator, dtype=F64)
    return STensor.from_torch(a), STensor.from_torch(b)


def _assert_refusal(error: BaseException, target: str) -> None:
    assert type(error) is UnsupportedFeature
    assert getattr(error, "code", None) == REFUSAL
    assert defect_code(error) == REFUSAL
    diagnostics = error.diagnostics  # type: ignore[attr-defined]
    assert [diagnostic.code for diagnostic in diagnostics] == [REFUSAL] * len(
        diagnostics
    )
    # One diagnostic, for the index the tile split: only it carries a sum.
    assert [diagnostic.message.split("'")[1] for diagnostic in diagnostics] == [target]
    assert diagnostics[0].entity_id is not None
    assert all(diagnostic.pass_name == "normalize_cin" for diagnostic in diagnostics)
    assert str(error).startswith(f"stage=normalized CIN: {REFUSAL} at root/")


# A tile of each index: i (outermost), k (the reduction) and j (innermost, the
# result's column; ken's run of main 2f01cd56 showed add_tile(j, 4, parallel=True,
# use_workspace=False) reaching the C++ compiler with j undeclared).
CASES = [
    (layout, target, parallel, use_workspace)
    for layout in ("no_ws", "dense_ws")
    for target in ("i", "j", "k")
    for parallel in (False, True)
    for use_workspace in (False, True)
]


def _case_id(case: Tuple[str, str, bool, bool]) -> str:
    layout, target, parallel, use_workspace = case
    return f"{layout}-tile_{target}-parallel_{parallel}-use_workspace_{use_workspace}"


def _copies_node_ids(case: Tuple[str, str, bool, bool]) -> bool:
    """``add_tile``'s workspace for a tile of the innermost ``j`` copies the producer's
    index variables with their node ids, so two distinct ``IndexVar`` objects share a
    ``NodeId``.  Normalization's structural preflight (``verify_cin_structure``)
    refuses that as ``duplicate_node_id`` before the tile check runs, on main too.
    It is a defect of ``add_tile`` of its own; the program is still refused before
    any lowering."""

    layout, target, _, use_workspace = case
    return layout == "no_ws" and target == "j" and use_workspace


def _assert_case_refused(
    error: BaseException, case: Tuple[str, str, bool, bool]
) -> None:
    if _copies_node_ids(case):
        assert type(error) is VerificationError
        assert defect_code(error) == "duplicate_node_id"
        return
    _assert_refusal(error, case[1])


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_the_direct_route_refuses_a_caller_tile(
    monkeypatch: pytest.MonkeyPatch, arm: int, case: Tuple[str, str, bool, bool]
) -> None:
    _set_arm(monkeypatch, arm)
    _forbid_builds(monkeypatch)
    with pytest.raises((UnsupportedFeature, VerificationError)) as caught:
        ops.lower_and_exec_cin(_tiled(*case), (N, M), *_operands())
    _assert_case_refused(caught.value, case)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_normalization_and_the_schedule_route_refuse_a_caller_tile(
    monkeypatch: pytest.MonkeyPatch, case: Tuple[str, str, bool, bool]
) -> None:
    _forbid_builds(monkeypatch)
    for entry in (
        normalize_cin,
        lambda cin: Scheduler.apply_schedule(cin, Schedule()),
        Scheduler.auto_schedule_plan,
    ):
        with pytest.raises((UnsupportedFeature, VerificationError)) as caught:
            entry(_tiled(*case))
        _assert_case_refused(caught.value, case)


def test_a_hand_built_tile_is_refused() -> None:
    """A ``TileSizeVar`` built by hand, as ``add_tile`` builds it."""

    i, j = IndexVar("i"), IndexVar("j")
    i_out, i_in = IndexVar("i_out"), IndexVar("i_in")
    i.expr = i_out + i_in
    TileSizeVar(i_out, i_in, size=4)
    c = TensorVar("C", fmt="dd", shape=(N, M), dtype=F64)
    a = TensorVar("A", fmt="dd", shape=(N, M), dtype=F64)
    program = ForAll(i_out, ForAll(i_in, ForAll(j, TensorAssign(c[i, j], a[i, j]))))
    with pytest.raises(UnsupportedFeature) as caught:
        normalize_cin(program)
    _assert_refusal(caught.value, "i")


@pytest.mark.parametrize(
    "case",
    [case for case in CASES if case[1] != "j" and not (case[1] == "k" and case[2])],
    ids=_case_id,
)
def test_the_legacy_lowerer_still_lowers_the_caller_tile(
    case: Tuple[str, str, bool, bool],
) -> None:
    """``CINLowerer.lower_IndexStmt`` copies the caller's tree with its tile fields, so
    the tile size is declared and the tiled index is bound.  (A parallel tile of the
    reduction ``k`` is refused there with ``parallel_reduction``.  A tile of ``j`` is
    left out: whether this lowerer places ``j``'s uses inside ``j_in`` is not what
    normalization's refusal decides.)"""

    target = case[1]
    cpp = lower_and_print(_tiled(*case))
    assert f"constexpr int kTile_{target} = 4;" in cpp
    assert f"int64_t {target} = {target}_out + {target}_in;" in cpp


def test_an_untiled_statement_still_normalizes() -> None:
    for layout in ("no_ws", "dense_ws"):
        program, _ = _program(layout)
        normalized = normalize_cin(program)
        assert normalized is not program
