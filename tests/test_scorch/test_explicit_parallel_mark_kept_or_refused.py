"""A loop marked parallel=True must run in parallel in the lowered kernel, or the
shipped lowering refuses the program with ``parallel_mark_dropped`` (an
``InvalidSchedule``) before any kernel is built.

Main lowered such programs serially and said nothing, in three ways.  The iteration
lattice lowers an inner loop from a copy it rebuilds without the mark
(``LatticePoint.get_simplified_cin``), so a mark on any loop under another one was
dropped.  ``mark_first_for_loop_parallel`` marks nothing when the loop must stay serial
(appending to a compressed result carries its position between iterations: a
compressed vector, or the rows of a Gustavson product through a sparse workspace,
post-merge item 23) or is a while loop (two compressed levels merged).  The Schedule
API marks its loop by name after lowering with the same helper, so
``Schedule(parallel_loop="i")`` over a COO operand went serial too.  Because any
explicit mark also switches off the automatic parallelization of the outer loop, each
of those kernels ran on one worker.

The check counts, per loop name, the loops of the lowered kernel that run in parallel,
after the last pass that marks one, and compares them with the ForAlls the statement
marks.  Kernels whose marks are kept are unchanged: those cases are computed with
several workers against a float64 reference at a relative 1e-12, and the pragma is in
the text.  Both automatic arms are run.  A tile the caller made with
``Scheduler.add_tile`` is refused earlier, at normalization (post-merge item 22), so
its arm difference is seen only through ``CINLowerer``
(``test_a_default_placed_tile_of_j_keeps_its_mark_in_arm_0``).
"""

from __future__ import annotations

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
    Where,
    Workspace,
)
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)
from scorch.compiler.diagnostics import (  # type: ignore[import-untyped]
    InvalidSchedule,
    UnsupportedFeature,
)
from scorch.compiler.scheduler import (  # type: ignore[import-untyped]
    Schedule,
    Scheduler,
)
from scorch.exceptions import defect_code  # type: ignore[import-untyped]
from scorch.ops import matmul  # type: ignore[import-untyped]
from tests.test_scorch.test_helpers import lower_and_print

REFUSAL = "parallel_mark_dropped"
F64 = torch.float64
SMALL = (7, 6, 5)
LARGE = (200, 150, 170)


def _set_arm(monkeypatch: pytest.MonkeyPatch, arm: int) -> None:
    monkeypatch.setenv("SCORCH_REGBLOCK", str(arm))
    assert CompileOptions.from_environment().scheduler.regblock_enabled == bool(arm)


def _forbid_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a refused program reached the kernel build")

    monkeypatch.setattr(ops, "_prepare_generated_kernel_build", forbidden)


def _rand(*shape: int, keep: float = 0.5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(16 + len(shape))
    values = torch.rand(*shape, generator=generator, dtype=F64)
    values[values < 1 - keep] = 0.0
    return values


def _wrap(values: torch.Tensor, fmt: str) -> STensor:
    """An operand built without a conversion kernel: dense from torch, ``ds`` from a
    torch CSR matrix, ``s`` and ``sdd`` from their components."""

    if set(fmt) == {"d"}:
        return STensor.from_torch(values.clone())
    if fmt == "ds":
        return STensor.from_torch(values.clone().to_sparse_csr())
    if fmt == "s":
        crd = values.nonzero().flatten()
        return STensor.from_components(
            tuple(values.shape),
            "s",
            [[torch.tensor([0, len(crd)]), crd]],
            values[crd].clone(),
        )
    if fmt == "sdd":
        stored = torch.tensor(
            [b for b in range(values.shape[0]) if bool(values[b].any())],
            dtype=torch.int64,
        )
        return STensor.from_components(
            tuple(values.shape),
            "sdd",
            [[torch.tensor([0, len(stored)]), stored], [], []],
            values[stored].reshape(-1).clone(),
        )
    raise ValueError(fmt)


def _matmul_vars(
    shape: Tuple[int, int, int], a_format: str, b_format: str
) -> Tuple[TensorVar, TensorVar, TensorVar]:
    n, k, m = shape
    c = TensorVar("C", fmt="dd", shape=(n, m), dtype=F64)
    a = TensorVar("A", fmt=a_format, shape=(n, k), dtype=F64)
    b = TensorVar("B", fmt=b_format, shape=(k, m), dtype=F64)
    return c, a, b


def _free_j(layout: str, shape: Tuple[int, int, int]) -> Any:
    """``C(i, j) += A(i, k) * B(k, j)``, dd x dd, j marked under i (and k)."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c, a, b = _matmul_vars(shape, "dd", "dd")
    assign = TensorAssign(c[i, j], a[i, k] * b[k, j], op=Operation.ADD)
    if layout == "inner":
        return ForAll(i, ForAll(k, ForAll(j, assign, parallel=True)))
    return ForAll(i, ForAll(j, ForAll(k, assign), parallel=True))


def _consumer_j(shape: Tuple[int, int, int]) -> Any:
    """ds x ds through a dense workspace; the consumer's loop over j is marked."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c, a, b = _matmul_vars(shape, "ds", "ds")
    w = Workspace(name="w", dim=1, dtype=F64, dense=True)
    producer = ForAll(
        k, ForAll(j, TensorAssign(w[j], a[i, k] * b[k, j], op=Operation.ADD))
    )
    consumer = ForAll(j, TensorAssign(c[i, j], w[j], op=Operation.ADD), parallel=True)
    return ForAll(i, Where(producer=producer, consumer=consumer))


def _gustavson_i(shape: Tuple[int, int, int]) -> Any:
    """``C = A @ B`` with all three ``ds``, row by row through a sparse workspace
    (Gustavson), the loop over i marked: appending row i to the compressed result
    carries C's position from one row to the next."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    n, kk, m = shape
    c = TensorVar("C", fmt="ds", shape=(n, m), dtype=F64)
    a = TensorVar("A", fmt="ds", shape=(n, kk), dtype=F64)
    b = TensorVar("B", fmt="ds", shape=(kk, m), dtype=F64)
    w = Workspace(name="w", dim=1, dtype=F64, dense=False)
    producer = ForAll(
        k, ForAll(j, TensorAssign(w[j], a[i, k] * b[k, j], op=Operation.ADD))
    )
    consumer = ForAll(j, TensorAssign(c[i, j], w[j]))
    return ForAll(i, Where(producer=producer, consumer=consumer), parallel=True)


def _sparse_result_i(n: int) -> Any:
    """``y(i) = a(i) * b(i)``, y and a compressed, i marked: appending to y carries
    y's position from one iteration to the next."""

    i = IndexVar("i")
    y = TensorVar("y", fmt="s", shape=(n,), dtype=F64)
    a = TensorVar("a", fmt="s", shape=(n,), dtype=F64)
    b = TensorVar("b", fmt="d", shape=(n,), dtype=F64)
    return ForAll(i, TensorAssign(y[i], a[i] * b[i]), parallel=True)


def _merge_b(batches: int, n: int, k: int, m: int) -> Any:
    """``C(b, i, k) += A(b, i, j) * B(b, j, k)``, b compressed in both operands and
    marked: the loop over b is a while loop that merges the two levels."""

    bv, i, j, kv = IndexVar("b"), IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="ddd", shape=(batches, n, m), dtype=F64)
    a = TensorVar("A", fmt="sdd", shape=(batches, n, k), dtype=F64)
    b = TensorVar("B", fmt="sdd", shape=(batches, k, m), dtype=F64)
    assign = TensorAssign(c[bv, i, kv], a[bv, i, j] * b[bv, j, kv], op=Operation.ADD)
    return ForAll(bv, ForAll(i, ForAll(j, ForAll(kv, assign))), parallel=True)


def _tile_j(shape: Tuple[int, int, int]) -> Any:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c, a, b = _matmul_vars(shape, "dd", "dd")
    assign = TensorAssign(c[i, j], a[i, k] * b[k, j], op=Operation.ADD)
    return Scheduler.add_tile(
        ForAll(i, ForAll(k, ForAll(j, assign))),
        j,
        4,
        parallel=True,
        use_workspace=False,
    )


A_SMALL, B_SMALL = _rand(SMALL[0], SMALL[1]), _rand(SMALL[1], SMALL[2], keep=0.6)
A3, B3 = _rand(3, SMALL[0], SMALL[1]), _rand(3, SMALL[1], SMALL[2], keep=0.6)
A3[1] = 0.0  # one batch stored in B only, so the merge over b skips a coordinate
VA, VB = _rand(SMALL[0]), _rand(SMALL[0], keep=0.9)

# (program, result shape, operands, the loop the refusal names, words of its reason)
RefusedCase = Tuple[Callable[[], Any], Tuple[int, ...], Callable[[], Any], str, str]
REFUSED: Dict[str, RefusedCase] = {
    "j under i and k": (
        lambda: _free_j("inner", SMALL),
        (SMALL[0], SMALL[2]),
        lambda: (_wrap(A_SMALL, "dd"), _wrap(B_SMALL, "dd")),
        "j",
        "copy of the loop that carries no mark",
    ),
    "j under i": (
        lambda: _free_j("middle", SMALL),
        (SMALL[0], SMALL[2]),
        lambda: (_wrap(A_SMALL, "dd"), _wrap(B_SMALL, "dd")),
        "j",
        "copy of the loop that carries no mark",
    ),
    "the consumer's j under i": (
        lambda: _consumer_j(SMALL),
        (SMALL[0], SMALL[2]),
        lambda: (_wrap(A_SMALL, "ds"), _wrap(B_SMALL, "ds")),
        "j",
        "copy of the loop that carries no mark",
    ),
    "i over Gustavson's compressed result": (
        lambda: _gustavson_i(SMALL),
        (SMALL[0], SMALL[2]),
        lambda: (_wrap(A_SMALL, "ds"), _wrap(B_SMALL, "ds")),
        "i",
        "must stay serial",
    ),
    "i over a compressed result": (
        lambda: _sparse_result_i(SMALL[0]),
        (SMALL[0],),
        lambda: (_wrap(VA, "s"), _wrap(VB, "d")),
        "i",
        "must stay serial",
    ),
    "b merged from two compressed levels": (
        lambda: _merge_b(3, *SMALL),
        (3, SMALL[0], SMALL[2]),
        lambda: (_wrap(A3, "sdd"), _wrap(B3, "sdd")),
        "b",
        "no OpenMP-compatible counted loop",
    ),
}


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("case", list(REFUSED))
def test_a_mark_the_kernel_does_not_keep_is_refused(
    monkeypatch: pytest.MonkeyPatch, arm: int, case: str
) -> None:
    _set_arm(monkeypatch, arm)
    build, result_shape, operands, loop, reason = REFUSED[case]
    tensors = operands()
    program = build()
    _forbid_builds(monkeypatch)
    with pytest.raises(InvalidSchedule) as caught:
        ops.lower_and_exec_cin(program, result_shape, *tensors)
    assert defect_code(caught.value) == REFUSAL
    assert [d.code for d in caught.value.diagnostics] == [REFUSAL]
    assert f"parallel_mark_dropped (loop {loop})" in str(caught.value)
    assert reason in caught.value.diagnostics[0].message


@pytest.mark.parametrize("arm", [0, 1])
def test_a_scheduled_parallel_loop_over_coo_rows_is_refused(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    """``Schedule(parallel_loop="i")`` with a COO operand: the marking declines the
    loop over i (it carries the position in A's row segments), so main computed
    serially.  The same schedule over a CSR operand keeps its mark
    (``test_a_scheduled_parallel_loop_over_csr_rows_stays_marked``)."""

    _set_arm(monkeypatch, arm)
    a, b = _rand(SMALL[0], SMALL[1]), _rand(SMALL[1], SMALL[2], keep=0.6)
    _forbid_builds(monkeypatch)
    with pytest.raises(InvalidSchedule) as caught:
        matmul(a.to_sparse(), b, use_cache=False, schedule=Schedule(parallel_loop="i"))
    assert defect_code(caught.value) == REFUSAL
    assert "parallel_mark_dropped (loop i)" in str(caught.value)
    assert "must stay serial" in caught.value.diagnostics[0].message


@pytest.mark.parametrize("arm", [0, 1])
def test_a_scheduled_parallel_loop_over_csr_rows_stays_marked(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    n, k, m = LARGE
    a, b = _rand(n, k), _rand(k, m, keep=0.6)
    got = matmul(
        a.to_sparse_csr(), b, use_cache=False, schedule=Schedule(parallel_loop="i")
    )
    got = got.to_torch() if isinstance(got, STensor) else got
    assert torch.allclose(got, a @ b, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("arm", [0, 1])
def test_a_default_placed_tile_of_j_is_refused_on_the_direct_route(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    """``Scheduler.add_tile(j, 4, parallel=True, use_workspace=False)`` gives the
    statement a tile its caller made.  Normalization refuses that with
    ``caller_tiled_cin_unsupported`` before anything is lowered (post-merge item 22),
    in both arms.  Main lowered it without its tile: in arm 1 the lattice dropped the
    mark on j_out, and in arm 0 the C++ used j outside the loop that declares it."""

    _set_arm(monkeypatch, arm)
    a, b = _wrap(A_SMALL, "dd"), _wrap(B_SMALL, "dd")
    program = _tile_j(SMALL)
    _forbid_builds(monkeypatch)
    with pytest.raises(UnsupportedFeature) as caught:
        ops.lower_and_exec_cin(program, (SMALL[0], SMALL[2]), a, b)
    assert defect_code(caught.value) == "caller_tiled_cin_unsupported"


def test_a_default_placed_tile_of_j_keeps_its_mark_in_arm_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CINLowerer.lower_IndexStmt`` does not normalize, so it still lowers the
    caller's tile.  Arm 0 puts j_out outermost, and the pragma stays on it, outside
    the loop over i.  (Arm 1 puts j_out under i, where the lattice drops the mark: a
    reported arm difference, to be removed by placing j_out where arm 0 does.)"""

    _set_arm(monkeypatch, 0)
    text = lower_and_print(_tile_j(SMALL))
    pragma = text.index("#pragma omp parallel for")
    j_out_loop = text.index("for (int64_t j_out", pragma)
    assert j_out_loop < text.index("for (int64_t i", pragma)


def test_a_statement_without_marks_records_none() -> None:
    """The check costs nothing on a statement that marks no loop: it records no
    marked ForAll, so it has nothing to count."""

    from scorch.compiler.cin_lowerer import CINLowerer  # type: ignore[import-untyped]

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c, a, b = _matmul_vars(SMALL, "dd", "dd")
    unmarked = ForAll(
        i,
        ForAll(
            k, ForAll(j, TensorAssign(c[i, j], a[i, k] * b[k, j], op=Operation.ADD))
        ),
    )
    assert CINLowerer._count_explicit_parallel_loops(unmarked, {}) == {}
    assert CINLowerer._count_explicit_parallel_loops(_free_j("inner", SMALL), {}) == {
        "j": 1
    }
