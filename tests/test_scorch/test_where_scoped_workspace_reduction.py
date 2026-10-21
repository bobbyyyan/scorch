"""A workspace declared inside ``ForAll(i)`` holds one value per ``i``.

The full CIN analysis (``cin_analysis._compute_cin_analysis``) took an index as reduced
into an assignment whenever a right-hand-side access used it and the left-hand side did
not.  ``ops.spmv``'s CIN accumulates ``A[i, j] * x[j]`` into a scalar workspace declared
by a ``Where`` inside ``ForAll(i)``, so ``i`` counted as reduced, and since ``i`` is
free in ``y[i]`` the verifier refused the program with ``free_reduction_conflict``.
Under full CIN verification (``SCORCH_VERIFY_CIN=1`` or ``full_cin_verification()``)
that refused every ``ops.spmv`` call and every ``scorch.matmul`` of a matrix by a vector
that reaches it: 22 of the 1308 programs of sapling job 83071, on the shipped compiler
and through LoopIR's entry, and no program of any other entry.

The rule now gives a workspace the indices of the ForAlls around the Where that declares
it, as leading indices of its own: each iteration of them fills and reads its own
instance.  An index the left-hand side carries is not reduced into it, and the consumer
that reads the workspace into a tensor without that index reduces it there.  A workspace
declared outside ``ForAll(i)`` carries no ``i``, so the conflict the rule exists for is
still refused.

LoopIR's entry then refused the same CIN for a second reason: it bound the runtime
result's shape and dtype onto every left-hand-side tensor, the workspace included, and
its own verification refused a scalar workspace of shape ``(7,)``.  It now binds them
onto the result alone, and LoopIR's lowering refuses the program next, by name: it
lowers no Where (``unsupported_statement``).  With a requested schedule the scheduler
ran first and planned the loops above the Where alone, which its plan verification
refused with no code; that route now meets the lowering's refusal before the scheduler.

Values are small integers, so float64 results are compared exactly.
"""

import itertools
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch

import scorch
from scorch import STensor
from scorch.compiler.cin import (
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
    Where,
    Workspace,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.cin_analysis import (
    CINAnalysis,
    full_cin_verification,
    verify_cin,
)
from scorch.compiler.diagnostics import VerificationError
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.pipeline import (
    _bind_runtime_metadata,
    execute_cin_via_loopir,
)
from scorch.compiler.scheduler import Schedule
from scorch.format import parse_format
from scorch.ops import spmv

F64, F32 = torch.float64, torch.float32
ROWS, COLS = 7, 5


@pytest.fixture(params=["0", "1"], ids=["regblock-off", "regblock-on"])
def arm(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("SCORCH_REGBLOCK", request.param)
    return request.param


# -- CINs built by hand ----------------------------------------------------------------


def _tensor(
    name: str,
    shape: Optional[Tuple[int, ...]],
    fmt: str,
    dtype: torch.dtype = F64,
) -> TensorVar:
    return TensorVar(name, shape=shape, fmt=parse_format(fmt), dtype=dtype)


def _codes(error: VerificationError) -> List[str]:
    return sorted({getattr(d, "code", None) for d in error.diagnostics} - {None})


def _ids(*index_vars: IndexVar) -> Tuple[Any, ...]:
    return tuple(index_var.index_id for index_var in index_vars)


def _reductions(analysis: CINAnalysis, lhs_name: str) -> Tuple[Any, ...]:
    """The reduction indices the analysis gives the assignment into ``lhs_name``."""

    (found,) = [
        info.reduction_index_ids
        for info in analysis.assignments.values()
        if analysis.symbol_definitions[
            analysis.accesses[info.lhs_access_id].tensor_id
        ].display_name
        == lhs_name
    ]
    return found


class _Program:
    """A CIN built by hand, with the tensors a test looks at."""

    def __init__(
        self,
        cin: Any,
        workspace: Workspace,
        result: TensorVar,
        free: Tuple[IndexVar, ...],
        reduced: Tuple[IndexVar, ...],
    ) -> None:
        self.cin, self.workspace, self.result = cin, workspace, result
        self.free, self.reduced = free, reduced


def _spmv_cin(
    a_fmt: str = "ds",
    x_fmt: str = "d",
    out_fmt: str = "d",
    workspace_dtype: torch.dtype = F64,
    result_shape: Optional[Tuple[int, ...]] = (ROWS,),
) -> _Program:
    """ops.spmv's CIN: y[i] = sum_j A[i, j] * x[j] through a scalar workspace."""

    i, j = IndexVar("i"), IndexVar("j")
    y = _tensor("y", result_shape, out_fmt)
    A, x = _tensor("A", (ROWS, COLS), a_fmt), _tensor("x", (COLS,), x_fmt)
    wksp = Workspace(name="wksp", dim=0, dtype=workspace_dtype)
    cin = ForAll(
        i,
        Where(
            producer=ForAll(
                j,
                TensorAssign(
                    wksp.get_default_access(), A[i, j] * x[j], op=Operation.ADD
                ),
            ),
            consumer=TensorAssign(y[i], wksp.get_default_access()),
        ),
    )
    return _Program(cin, wksp, y, (i,), (j,))


def _row_workspace_cin() -> _Program:
    """A row of C = A @ B accumulated in w[k], a workspace declared inside ForAll(i)."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    C = _tensor("C", (ROWS, 3), "dd")
    A, B = _tensor("A", (ROWS, COLS), "ds"), _tensor("B", (COLS, 3), "dd")
    w = Workspace(name="w", dim=1, dtype=F64, dense=True)
    cin = ForAll(
        i,
        Where(
            producer=ForAll(
                j,
                ForAll(k, TensorAssign(w[k], A[i, j] * B[j, k], op=Operation.ADD)),
            ),
            consumer=ForAll(k, TensorAssign(C[i, k], w[k])),
        ),
    )
    return _Program(cin, w, C, (i, k), (j,))


def test_a_scalar_workspace_inside_forall_i_holds_one_value_per_i() -> None:
    program = _spmv_cin()
    analysis = verify_cin(program.cin)
    assert analysis.free_index_ids == _ids(*program.free)
    assert analysis.reduction_index_ids == _ids(*program.reduced)
    assert _reductions(analysis, "wksp") == _ids(*program.reduced)
    assert _reductions(analysis, "y") == ()


def test_a_vector_workspace_inside_forall_i_holds_one_row_per_i() -> None:
    program = _row_workspace_cin()
    analysis = verify_cin(program.cin)
    assert set(analysis.free_index_ids) == set(_ids(*program.free))
    assert analysis.reduction_index_ids == _ids(*program.reduced)
    assert _reductions(analysis, "w") == _ids(*program.reduced)


# LoopIR's entry bound the runtime result's shape and dtype onto every left-hand side,
# the workspace included (get_result_tensor_vars names both): the scalar workspace got
# shape (7,), which its own verification refused (tensor_shape_rank_mismatch, sapling
# job 83129), and a workspace declared in another type than the output's took the
# output's, which changes the precision it accumulates in.
_BINDINGS = {
    "scalar workspace": (
        lambda: _spmv_cin(result_shape=None),
        [((ROWS, COLS), F64), ((COLS,), F64)],
        (ROWS,),
        F64,
    ),
    "rank-1 workspace": (
        _row_workspace_cin,
        [((ROWS, COLS), F64), ((COLS, 3), F64)],
        (ROWS, 3),
        F64,
    ),
    "float64 workspace, float32 operands": (
        lambda: _spmv_cin(result_shape=None),
        [((ROWS, COLS), F32), ((COLS,), F32)],
        (ROWS,),
        F32,
    ),
}


@pytest.mark.parametrize("case", list(_BINDINGS))
def test_loopir_binds_the_runtime_result_onto_the_result_alone(case: str) -> None:
    build, inputs, result_shape, output_dtype = _BINDINGS[case]
    program = build()
    declared = (program.workspace.shape, program.workspace.dtype)
    _bind_runtime_metadata(program.cin, inputs, result_shape)
    assert (program.workspace.shape, program.workspace.dtype) == declared
    assert program.result.shape == result_shape
    assert program.result.dtype == output_dtype


def test_a_consumer_that_drops_an_enclosing_index_reduces_it() -> None:
    # y[i] = sum_k A[i, k] with the workspace declared inside ForAll(k): the producer
    # reduces nothing (each k has its own instance), the consumer reduces k.  The old
    # rule read i and k as reduced into the workspace and refused i, which is free.
    i, k = IndexVar("i"), IndexVar("k")
    y = _tensor("y", (ROWS,), "d")
    A = _tensor("A", (ROWS, COLS), "dd")
    wksp = Workspace(name="wksp", dim=0, dtype=F64)
    cin = ForAll(
        i,
        ForAll(
            k,
            Where(
                producer=TensorAssign(wksp.get_default_access(), A[i, k]),
                consumer=TensorAssign(
                    y[i], wksp.get_default_access(), op=Operation.ADD
                ),
            ),
        ),
    )
    analysis = verify_cin(cin)
    assert analysis.free_index_ids == _ids(i)
    assert analysis.reduction_index_ids == _ids(k)
    assert _reductions(analysis, "wksp") == ()
    assert _reductions(analysis, "y") == _ids(k)


def test_a_workspace_declared_outside_forall_i_and_summed_over_i_is_refused() -> None:
    # Negative control: one scalar for every row, summed over i and j, read into y[i].
    # i is free in the result and reduced into the workspace, which the rule refuses.
    i, j = IndexVar("i"), IndexVar("j")
    y = _tensor("y", (ROWS,), "d")
    A, x = _tensor("A", (ROWS, COLS), "ds"), _tensor("x", (COLS,), "d")
    wksp = Workspace(name="wksp", dim=0, dtype=F64)
    cin = Where(
        producer=ForAll(
            i,
            ForAll(
                j,
                TensorAssign(
                    wksp.get_default_access(), A[i, j] * x[j], op=Operation.ADD
                ),
            ),
        ),
        consumer=ForAll(i, TensorAssign(y[i], wksp.get_default_access())),
    )
    with pytest.raises(VerificationError) as error:
        verify_cin(cin)
    assert "free_reduction_conflict" in _codes(error.value)


def test_an_index_reduced_and_free_in_the_result_is_still_refused() -> None:
    # Negative control: j is reduced by the ForAll inside the producer, and free in
    # Y[i, j].
    i, j = IndexVar("i"), IndexVar("j")
    Y = _tensor("Y", (ROWS, COLS), "dd")
    A = _tensor("A", (ROWS, COLS), "dd")
    wksp = Workspace(name="wksp", dim=0, dtype=F64)
    cin = ForAll(
        i,
        Where(
            producer=ForAll(
                j, TensorAssign(wksp.get_default_access(), A[i, j], op=Operation.ADD)
            ),
            consumer=ForAll(j, TensorAssign(Y[i, j], wksp.get_default_access())),
        ),
    )
    with pytest.raises(VerificationError) as error:
        verify_cin(cin)
    assert "free_reduction_conflict" in _codes(error.value)


# -- the census programs, through the real entries -------------------------------------


def _dense_matrix() -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    values = torch.randint(1, 9, (ROWS, COLS), generator=generator).to(F64)
    keep = torch.rand(ROWS, COLS, generator=generator) < 0.5
    keep[2] = False
    return values * keep


def _components(
    physical: torch.Tensor, fmt: str, index: torch.dtype
) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
    """``physical`` stored in ``fmt`` with torch alone (no conversion kernel).

    Every index array is made contiguous: a column of ``nonzero()`` or of the paths is
    a strided view, ``.to()`` of the dtype it already has returns it as it is, and
    ``from_components`` refuses a non-contiguous index array.
    """

    if set(fmt) == {"o"}:
        stored = physical.nonzero()
        coordinates = [
            [stored[:, level].to(index).contiguous()] for level in range(len(fmt))
        ]
        return coordinates, physical[tuple(stored.T)]
    nonzero = physical != 0
    paths = torch.zeros((1, 0), dtype=torch.int64)
    mode_indices: List[List[torch.Tensor]] = []
    for level, kind in enumerate(fmt):
        extent = physical.shape[level]
        parents = paths.shape[0]
        children = torch.cat(
            [
                paths.repeat_interleave(extent, dim=0),
                torch.arange(extent).repeat(parents).unsqueeze(1),
            ],
            dim=1,
        )
        if kind == "d":
            mode_indices.append([])
            paths = children
            continue
        below = nonzero
        for _ in range(physical.dim() - level - 1):
            below = below.any(dim=-1)
        keep = below[tuple(children.T)]
        positions = torch.zeros(parents + 1, dtype=torch.int64)
        positions[1:] = torch.cumsum(keep.reshape(parents, extent).sum(dim=1), 0)
        paths = children[keep]
        mode_indices.append([positions.to(index), paths[:, -1].to(index).contiguous()])
    return mode_indices, physical[tuple(paths.T)]


def _stored(
    dense: torch.Tensor,
    fmt: str,
    mode_order: Optional[Sequence[int]],
    name: str,
    index: torch.dtype = torch.int64,
) -> STensor:
    order = list(range(dense.dim())) if mode_order is None else list(mode_order)
    physical = dense.permute(*order).contiguous()
    mode_indices, values = _components(physical, fmt, index)
    return STensor.from_components(
        tuple(physical.shape),
        fmt,
        mode_indices,
        values,
        name=name,
        mode_order=order,
        index_dtype=index,
    )


def _logical(result: Any) -> torch.Tensor:
    return (
        result if isinstance(result, torch.Tensor) else result.to_torch(in_place=False)
    )


VECTOR = torch.tensor([2.0, 0.0, -1.0, 3.0, 4.0], dtype=F64)
SPMV_CASES = list(
    itertools.product(["ds", "ss"], [None, (1, 0)], ["d", "s"], [None, "s"])
)


@pytest.mark.parametrize(
    "fmt, stored, x_fmt, output_format",
    SPMV_CASES,
    ids=[f"A-{a}-{s}-x-{x}-out-{o}" for a, s, x, o in SPMV_CASES],
)
def test_spmv_passes_full_cin_verification(
    arm: str,
    fmt: str,
    stored: Optional[Tuple[int, int]],
    x_fmt: str,
    output_format: Optional[str],
) -> None:
    dense = _dense_matrix()
    a = _stored(dense, fmt, stored, "A")
    x = _stored(VECTOR, x_fmt, None, "x")
    with full_cin_verification():
        result = spmv(a, x, output_format=output_format)
    assert torch.equal(_logical(result), dense @ VECTOR)


MATMUL_CASES = list(
    itertools.product(["ds", "sd", "ss", "oo"], [None, (1, 0)], [True, False])
)


@pytest.mark.parametrize(
    "fmt, stored, use_cache",
    MATMUL_CASES,
    ids=[f"A-{a}-{s}-cache-{c}" for a, s, c in MATMUL_CASES],
)
def test_matmul_of_a_matrix_by_a_vector_passes_full_cin_verification(
    arm: str, fmt: str, stored: Optional[Tuple[int, int]], use_cache: bool
) -> None:
    dense = _dense_matrix()
    a = _stored(dense, fmt, stored, "A")
    with full_cin_verification():
        result = scorch.matmul(a, VECTOR.clone(), use_cache=use_cache)
    assert torch.equal(_logical(result), dense @ VECTOR)


# -- the same CINs through LoopIR's entry, without verification ------------------------

# The census's 22 programs build ops.spmv's CIN for A in ds, ss, sd and oo (the matmul
# ones reach ops.spmv with A as given), x in d or s, and a result in d or s.  LoopIR's
# entry runs each one, without full CIN verification, on both automatic arms: the result
# must equal the float64 product, or the entry must refuse the program with the code
# named here.
#
# Every one is refused today, by name, on both of the entry's routes:
# - without a requested schedule, past the runtime binding: lower_cin._collect_loop_nest
#   raises LoopIRLoweringError ``unsupported_statement`` ("Where/workspace programs are
#   outside the dense LoopIR families").  Before the binding fix the same programs were
#   refused earlier, with ``tensor_shape_rank_mismatch`` on the workspace;
# - with one, the same refusal before the scheduler (lower_cin.refuse_where_program).
#   Without it the scheduler's automatic plan ordered the loops above the Where alone,
#   and its plan verification refused the plan with no code.
# A LoopIR that lowers a Where moves its cases to "correct" here.
LOOPIR_CASES = list(itertools.product(["ds", "ss", "sd", "oo"], ["d", "s"], ["d", "s"]))
LOOPIR_REFUSALS: Dict[Tuple[str, str, str], str] = {
    case: "unsupported_statement" for case in LOOPIR_CASES
}
# The function each route's refusal is raised in.
LOOPIR_ROUTES: Dict[str, Tuple[Optional[Schedule], str]] = {
    "no-schedule": (None, "_collect_loop_nest"),
    "automatic-schedule": (Schedule(), "refuse_where_program"),
}


def _refusal_code(error: Exception) -> str:
    defect = getattr(error, "defect", None)
    if getattr(defect, "code", None):
        return str(defect.code)
    if getattr(error, "diagnostics", None):
        return ",".join(_codes(error))  # type: ignore[arg-type]
    code = getattr(error, "code", None)
    return str(code) if code else f"unstructured {type(error).__name__}: {error}"


def _raised_in(error: BaseException) -> str:
    """The function the exception was raised from, below lower_cin._fail."""

    names = []
    trace = error.__traceback__
    while trace is not None:
        names.append(trace.tb_frame.f_code.co_name)
        trace = trace.tb_next
    while names and names[-1] == "_fail":
        names.pop()
    return names[-1] if names else ""


@pytest.mark.parametrize("route", list(LOOPIR_ROUTES))
@pytest.mark.parametrize("regblock", [False, True], ids=["regblock-off", "regblock-on"])
@pytest.mark.parametrize(
    "a_fmt, x_fmt, out_fmt",
    LOOPIR_CASES,
    ids=[f"A-{a}-x-{x}-out-{o}" for a, x, o in LOOPIR_CASES],
)
def test_loopir_runs_the_spmv_cin_or_refuses_it_by_name(
    route: str, regblock: bool, a_fmt: str, x_fmt: str, out_fmt: str
) -> None:
    dense = _dense_matrix()
    a = _stored(dense, a_fmt, None, "A", torch.int32)
    x = _stored(VECTOR, x_fmt, None, "x", torch.int32)
    program = _spmv_cin(a_fmt, x_fmt, out_fmt)
    expected = LOOPIR_REFUSALS.get((a_fmt, x_fmt, out_fmt), "correct")
    schedule, refused_in = LOOPIR_ROUTES[route]
    options = CompileOptions.from_environment().with_regblock_enabled(regblock)
    if schedule is not None:
        options = replace(options, requested_schedule=schedule)
    try:
        result = execute_cin_via_loopir(
            program.cin, (ROWS,), a, x, compile_options=options
        )
    except Exception as error:  # noqa: BLE001 -- its code is the assertion
        assert _refusal_code(error) == expected, error
        if expected == "unsupported_statement":
            # The lowering's own refusal: without a schedule, after the runtime binding
            # accepted the program; with one, before the scheduler plans it.
            assert isinstance(error, LoopIRLoweringError), error
            assert _raised_in(error) == refused_in, error
        return
    if isinstance(result, tuple):
        result = result[0]
    assert expected == "correct", "LoopIR ran a program it is expected to refuse"
    assert torch.equal(_logical(result), dense @ VECTOR)
