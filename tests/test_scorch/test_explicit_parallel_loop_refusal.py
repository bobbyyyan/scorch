"""A loop marked parallel by hand on the direct CIN route must write cells its own
variable selects; the shipped lowering refuses any other mark, in both automatic arms,
with ``parallel_reduction`` (an ``InvalidSchedule``), before it emits anything.

``ForAll(k, ..., parallel=True)`` over a reduction variable ``k`` of
``C(i, j) += A(i, k) * B(k, j)`` used to be lowered to ``#pragma omp parallel for`` on
the loop over ``k``, whose iterations all add into every cell of C with no atomic and no
reduction clause: workers lost each other's updates.  The Schedule API refuses such a
mark before it sets it, and the LoopPlan route refuses it, both with
``parallel_reduction`` (``test_parallel_mark_rules.py``); the direct route checked
nothing.  The lowering
now checks every marked loop of the statement before lowering it: each tensor the loop
body writes must be indexed by the loop's variable, or by the variable a tile split into
the loop and its partner.  A workspace that a Where inside the loop produces is private
to an iteration and is not checked; a loop marked inside a producer is checked against
that producer's workspace.

Refused: the reduction loop outermost and marked, with no workspace, a dense workspace
or a sparse one; the reduction loop marked inside the row loop; a reduction loop marked
inside a producer; the outer loop of a tile of the reduction variable, which the direct
route now refuses earlier, at normalization, as every tile its caller made
(``caller_tiled_cin_unsupported``, ``test_caller_tiled_cin_refusal.py``), so it is
checked here through ``CINLowerer.lower_IndexStmt``, which keeps the tile.  Main lowered a
mark on an inner loop serially, because the iteration lattice rebuilds inner loops
without their mark; those programs are refused now, as the other two routes refuse
them.  Kept: a mark on the row loop, with no workspace and with a dense workspace (those
kernels still carry the pragma), and the Schedule API's mark on the outer loop of a tile
of the row, which reaches the check and passes it.  Each is computed with several
workers.  Values are compared with a float64 reference to a relative 1e-12.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pytest
import torch

from scorch import STensor, ops  # type: ignore[import-untyped]
from scorch.compiler.cin import (  # type: ignore[import-untyped]
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
    Workspace,
)
from scorch.compiler.cin_lowerer import CINLowerer  # type: ignore[import-untyped]
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    InvalidSchedule,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    Schedule,
    Scheduler,
    TileSpec,
)
from scorch.exceptions import defect_code  # type: ignore[import-untyped]
from scorch.ops import matmul  # type: ignore[import-untyped]
from tests.test_scorch.test_helpers import lower_and_print

REFUSAL = "parallel_reduction"
F64 = torch.float64
SMALL = (7, 6, 5)
LARGE = (200, 150, 170)


def _set_arm(monkeypatch: pytest.MonkeyPatch, arm: int) -> None:
    monkeypatch.setenv("SCORCH_REGBLOCK", str(arm))
    assert CompileOptions.from_environment().scheduler.regblock_enabled == bool(arm)


def _operands(
    a_format: str, b_format: str, shape: Tuple[int, int, int]
) -> Tuple[STensor, STensor, torch.Tensor]:
    n, k, m = shape
    generator = torch.Generator().manual_seed(16)
    a = torch.rand(n, k, generator=generator, dtype=F64)
    a[a < 0.5] = 0.0
    b = torch.rand(k, m, generator=generator, dtype=F64)
    b[b < 0.4] = 0.0

    def wrap(values: torch.Tensor, fmt: str) -> STensor:
        tensor = STensor.from_torch(values.clone())
        return tensor.to_dense() if fmt == "dd" else tensor.to_sparse(fmt)

    return wrap(a, a_format), wrap(b, b_format), a @ b


def _program(
    layout: str,
    a_format: str,
    b_format: str,
    workspace: Optional[str],
    shape: Tuple[int, int, int],
) -> Any:
    """``C(i, j) += A(i, k) * B(k, j)``, C ``dd``; ``workspace`` None (no Where),
    "dense" or "sparse" (``w(j) += A(i, k) * B(k, j)``, ``C(i, j) += w(j)``)."""

    n, k_extent, m = shape
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dd", shape=(n, m), dtype=F64)
    a = TensorVar("A", fmt=a_format, shape=(n, k_extent), dtype=F64)
    b = TensorVar("B", fmt=b_format, shape=(k_extent, m), dtype=F64)
    product = a[i, k] * b[k, j]
    if workspace is None:
        body: Any = ForAll(j, TensorAssign(c[i, j], product, op=Operation.ADD))
    else:
        w = Workspace(name="w", dim=1, dtype=F64, dense=workspace == "dense")
        producer: Any = ForAll(j, TensorAssign(w[j], product, op=Operation.ADD))
        if layout == "k-in-producer":
            producer = ForAll(k, producer, parallel=True)
        body = Where(
            producer=producer,
            consumer=ForAll(j, TensorAssign(c[i, j], w[j], op=Operation.ADD)),
        )
    if layout == "k-outer":
        return ForAll(k, ForAll(i, body), parallel=True)
    if layout == "k-inner":
        return ForAll(i, ForAll(k, body, parallel=True))
    if layout == "k-in-producer":
        return ForAll(i, body)
    if layout == "i-marked":
        return ForAll(i, ForAll(k, body), parallel=True)
    if layout == "tile-k":
        return Scheduler.add_tile(
            ForAll(i, ForAll(k, body)),
            k,
            4,
            placement="outermost",
            parallel=True,
            use_workspace=False,
        )
    raise ValueError(layout)


def _forbid_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a refused program reached the kernel build")

    monkeypatch.setattr(ops, "_prepare_generated_kernel_build", forbidden)


def _dense(result: Any) -> torch.Tensor:
    return result.to_torch() if isinstance(result, STensor) else result


REFUSED: Dict[str, Tuple[str, str, str, Optional[str], str]] = {
    # id: (layout, A format, B format, workspace, the loop the refusal names)
    "reduction outermost, no workspace": ("k-outer", "dd", "dd", None, "k"),
    "reduction outermost, dense workspace": ("k-outer", "ds", "ds", "dense", "k"),
    "reduction outermost, sparse workspace": ("k-outer", "ds", "ds", "sparse", "k"),
    "reduction inside the row loop, no workspace": ("k-inner", "dd", "dd", None, "k"),
    "reduction inside the row loop, sparse workspace": (
        "k-inner",
        "ds",
        "ds",
        "sparse",
        "k",
    ),
    "reduction inside a producer": ("k-in-producer", "ds", "ds", "dense", "k"),
}


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("case", list(REFUSED))
def test_a_marked_loop_that_shares_cells_is_refused(
    monkeypatch: pytest.MonkeyPatch, arm: int, case: str
) -> None:
    _set_arm(monkeypatch, arm)
    layout, a_format, b_format, workspace, loop = REFUSED[case]
    a, b, _ = _operands(a_format, b_format, SMALL)
    program = _program(layout, a_format, b_format, workspace, SMALL)
    _forbid_builds(monkeypatch)
    with pytest.raises(InvalidSchedule) as caught:
        ops.lower_and_exec_cin(program, (SMALL[0], SMALL[2]), a, b)
    assert defect_code(caught.value) == REFUSAL
    assert [d.code for d in caught.value.diagnostics] == [REFUSAL]
    written = "w" if layout == "k-in-producer" else "C"
    assert f"parallel_reduction for {written!r} (loop {loop})" in str(caught.value)


@pytest.mark.parametrize("arm", [0, 1])
def test_the_outer_loop_of_a_tile_of_the_reduction_is_refused(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    program = _program("tile-k", "dd", "dd", None, SMALL)
    _forbid_builds(monkeypatch)
    with pytest.raises(InvalidSchedule) as caught:
        CINLowerer().lower_IndexStmt(program)
    assert defect_code(caught.value) == REFUSAL
    assert [d.code for d in caught.value.diagnostics] == [REFUSAL]
    assert "parallel_reduction for 'C' (loop k_out)" in str(caught.value)


KEPT: Dict[str, Tuple[str, str, str, Optional[str]]] = {
    "row loop, no workspace": ("i-marked", "dd", "dd", None),
    "row loop, dense workspace": ("i-marked", "ds", "ds", "dense"),
}


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("case", list(KEPT))
def test_a_marked_loop_over_its_own_cells_stays_marked(
    monkeypatch: pytest.MonkeyPatch, arm: int, case: str
) -> None:
    """200 rows: the kernel asks for min(rows / 16, the thread ceiling) workers (and
    at most A's entries / 500 when A is compressed), so several workers run."""

    _set_arm(monkeypatch, arm)
    layout, a_format, b_format, workspace = KEPT[case]

    def build() -> Any:
        return _program(layout, a_format, b_format, workspace, LARGE)

    assert "#pragma omp parallel" in lower_and_print(build())
    a, b, want = _operands(a_format, b_format, LARGE)
    got = _dense(ops.lower_and_exec_cin(build(), (LARGE[0], LARGE[2]), a, b))
    assert got.shape == want.shape
    assert torch.allclose(got, want, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("arm", [0, 1])
def test_a_scheduled_tile_of_the_row_passes_the_check(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    """The Schedule API marks the outer loop of a tile of the row, ``i_out``, whose
    iterations cover their own rows: the check sees that loop and admits it."""

    _set_arm(monkeypatch, arm)
    checked: list[str] = []
    check = CINLowerer._refuse_parallel_loop_with_shared_cells

    def recording_check(lowerer: CINLowerer, stmt: ForAll) -> None:
        checked.append(stmt.get_index_var().name)
        check(lowerer, stmt)

    monkeypatch.setattr(
        CINLowerer, "_refuse_parallel_loop_with_shared_cells", recording_check
    )
    n, k, m = LARGE
    generator = torch.Generator().manual_seed(16)
    a = torch.rand(n, k, generator=generator, dtype=F64)
    a[a < 0.5] = 0.0
    b = torch.rand(k, m, generator=generator, dtype=F64)
    schedule = Schedule(
        loop_order=("i", "j", "k"),
        tiles=(
            TileSpec(
                index_var="i",
                width=4,
                placement="outermost",
                parallel=True,
                accum="direct",
                unroll=False,
            ),
        ),
        tag="parallel-row-i-t4",
    )
    got = _dense(matmul(a.to_sparse_csr(), b, use_cache=False, schedule=schedule))
    assert checked == ["i_out"]
    assert torch.allclose(got, a @ b, rtol=1e-12, atol=0.0)
