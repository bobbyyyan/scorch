"""Patch 0055: a scalar workspace's sparse result keeps every entry its loops visit.

``spmv`` into an ``"s"`` result (and ``scorch.matmul(A, x, format="s")``) lowers a
``Where`` whose scalar workspace sums one row.  Its consumer appended the row only when
the sum was nonzero, so a row that ``A`` stores was dropped when ``x`` is 0 at its
columns, when its products cancel, or when ``A`` stores an explicit 0 there.  The rule
(Bobby, 2026-10-12): a kernel keeps an explicit zero at a cell its loops visit.  The
consumer now appends when the producer reached the accumulation
(``scalar_workspace_visit.py``).

1. Kernel text, nothing built: no sum-against-zero test guards an append; the visit
   flag guards it, set before the producer loop where every iteration accumulates and
   in the matched branch where a merge decides.  A dense result keeps its guard.
2. ``place_visit_flag`` on hand-built loops: the entry test, and each case it leaves
   in the loop.
3. Execution: exact stored rows and values against the stored structure, every format
   of ``A``, dense and sparse ``x``, both automatic arms, float64 and int32.
   mlsys-submission's four failing cases (``A`` in ``"ss"`` and ``"oo"``, dense ``x``,
   output ``"s"``, both arms; sapling jobs 82871 and 82901) are among them, through
   ``spmv`` and through ``matmul``.  The new compiler refuses the same program by name
   (its lowering lowers no ``Where``); the refusal is pinned until it computes it.
"""

import re
from typing import Dict, List, Tuple

import pytest
import torch

import scorch
from scorch import ops, stensor
from scorch.compiler import llir
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
from scorch.compiler.llir import AssignOp
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.pipeline import execute_cin_via_loopir
from scorch.compiler.scalar_workspace_visit import (
    place_visit_flag,
    visit_flag,
    visit_flag_set,
)
from scorch.stensor import STensor
from tests.test_scorch.test_sparse_assembly_strategy import auto_options

ROWS, COLUMNS = 7, 5

# The stored entries of A, (row, column, value), with each way a visited row sums to 0.
_ENTRIES = [
    (0, 0, 1),
    (0, 2, 2),  # row 0: a nonzero sum; row 1 stores nothing
    (2, 3, 4),  # row 2: x is 0 at column 3 (sparse x stores an explicit 0 there)
    (3, 0, 1),
    (3, 1, -1),  # row 3: two products that cancel
    (4, 4, 3),  # row 4: a nonzero sum
    (5, 2, 0),  # row 5: A stores an explicit 0 (sparse x does not store column 2)
    (6, 1, 5),
    (6, 4, 1),
]
_X_DENSE = [1, 1, 3, 0, 2]
_X_STORED = {0: 1, 1: 1, 3: 0, 4: 2}

_FORMATS = ("ds", "ss", "sd", "dd", "oo")
_INDEX = torch.int32


def _matrix(fmt: str, dtype) -> Tuple[STensor, List[Tuple[int, int, int]]]:
    """A in ``fmt`` and its stored cells, (row, column, value): a dense level stores
    every cell of its fiber, zeros included."""

    dense = torch.zeros(ROWS, COLUMNS, dtype=torch.int64)
    for row, column, value in _ENTRIES:
        dense[row, column] = value
    stored_rows = sorted({row for row, _, _ in _ENTRIES})
    if fmt == "dd":
        cells = [(i, j, int(dense[i, j])) for i in range(ROWS) for j in range(COLUMNS)]
        levels: List[List[torch.Tensor]] = [[], []]
        values = dense.flatten()
    elif fmt == "sd":
        cells = [(i, j, int(dense[i, j])) for i in stored_rows for j in range(COLUMNS)]
        levels = [[torch.tensor([0, len(stored_rows)]), torch.tensor(stored_rows)], []]
        values = dense[stored_rows].flatten()
    else:
        cells = list(_ENTRIES)
        values = torch.tensor([value for _, _, value in cells])
        columns = torch.tensor([column for _, column, _ in cells])
        if fmt == "oo":
            levels = [[torch.tensor([row for row, _, _ in cells])], [columns]]
        else:
            counts = [sum(1 for row, _, _ in cells if row == i) for i in range(ROWS)]
            if fmt == "ds":
                parents = counts
                outer: List[torch.Tensor] = []
            else:
                parents = [counts[i] for i in stored_rows]
                outer = [torch.tensor([0, len(stored_rows)]), torch.tensor(stored_rows)]
            positions = torch.cumsum(torch.tensor([0] + parents), 0)
            levels = [outer, [positions, columns]]
    levels = [[array.to(_INDEX) for array in level] for level in levels]
    matrix = STensor.from_components(
        (ROWS, COLUMNS), fmt, levels, values.to(dtype), name="A"
    )
    return matrix, cells


def _vector(kind: str, dtype) -> Tuple[STensor, Dict[int, int]]:
    """x, dense or sparse, and its stored entries by column."""

    if kind == "d":
        stored = dict(enumerate(_X_DENSE))
        vector = STensor.from_components(
            (COLUMNS,), "d", [[]], torch.tensor(_X_DENSE).to(dtype), name="x"
        )
        return vector, stored
    stored = dict(_X_STORED)
    columns = sorted(stored)
    levels = [[torch.tensor([0, len(columns)]), torch.tensor(columns)]]
    vector = STensor.from_components(
        (COLUMNS,),
        "s",
        [[array.to(_INDEX) for array in level] for level in levels],
        torch.tensor([stored[column] for column in columns]).to(dtype),
        name="x",
    )
    return vector, stored


def _expected(cells, stored_x) -> Tuple[List[int], List[float]]:
    """The rows the producer visits (a stored cell of A whose column x stores) and
    their sums, in float64."""

    sums: Dict[int, float] = {}
    for row, column, value in cells:
        if column in stored_x:
            sums[row] = sums.get(row, 0.0) + float(value) * float(stored_x[column])
    rows = sorted(sums)
    return rows, [sums[row] for row in rows]


def _stored(result, fmt: str) -> Tuple[List[int], torch.Tensor]:
    assert isinstance(result, STensor), type(result)
    assert str(result.format).replace(",", "") == fmt, result.format
    level = result.storage.index.mode_indices[0]
    if fmt == "s":
        positions, coordinates = level
        assert positions.tolist() == [0, coordinates.numel()]
    else:
        (coordinates,) = level
    return coordinates.tolist(), result.storage.value


def _assert_result(result, fmt, cells, stored_x, dtype):
    rows, sums = _expected(cells, stored_x)
    got_rows, values = _stored(result, fmt)
    assert got_rows == rows, (got_rows, rows)
    assert values.dtype == dtype
    expected = torch.tensor(sums, dtype=torch.float64)
    assert torch.equal(values.to(torch.float64), expected), (values, expected)


def _set_arm(monkeypatch, arm: int) -> None:
    monkeypatch.setenv("SCORCH_REGBLOCK", str(arm))
    assert CompileOptions.from_environment().scheduler.regblock_enabled == bool(arm)


# --------------------------------------------------------------------------------------
# 1. Kernel text, nothing built.
# --------------------------------------------------------------------------------------


class _Captured(Exception):
    pass


def _spmv_source(monkeypatch, fmt: str, kind: str, output: str) -> str:
    captured: List[str] = []

    def loader(prepared):
        captured.append(prepared.request.cpp_sources[-1])
        raise _Captured()

    # spmv looks in the process's generated-kernel store before it builds, so a kernel an
    # earlier test of this process kept for the same call and options (section 3's
    # test_spmv_keeps_every_visited_row with SCORCH_REGBLOCK=0) would come back without
    # reaching the loader.
    monkeypatch.setattr(stensor, "_GENERATED_KERNELS", {})
    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", loader)
    matrix, _ = _matrix(fmt, torch.float64)
    vector, _ = _vector(kind, torch.float64)
    with pytest.raises(_Captured):
        ops.spmv(matrix, vector, output_format=output)
    (source,) = captured
    return source


def _evaluate_body(source: str) -> str:
    return source[source.rindex("evaluate(") :]


@pytest.mark.parametrize("output", ["s", "o"])
@pytest.mark.parametrize("kind", ["d", "s"])
@pytest.mark.parametrize("fmt", _FORMATS)
def test_the_append_is_guarded_by_the_visit_not_the_sum(monkeypatch, fmt, kind, output):
    body = _evaluate_body(_spmv_source(monkeypatch, fmt, kind, output))
    assert "wksp != 0" not in body
    assert "bool wksp_visited = false;" in body
    assert re.search(r"if \(wksp_visited\) \{\n\s*y0_crd\.push_back\(i\);", body), body
    # Set before the loop: only plain statements (a hoisted row pointer, for a dense A) may
    # stand between the flag and the loop it tests.
    before = re.findall(
        r"wksp_visited \|= [^\n]*;\n(?:[ \t]*[^\n{}]*;\n)*[ \t]*for \(", body
    )
    inside = re.findall(r"wksp \+= .*;\n\s*wksp_visited = true;", body)
    assert len(before) + len(inside) >= 1, body
    if kind == "d":
        # A dense x never decides a visit: A's own loop drives, every iteration
        # accumulates, and the flag is its entry test.
        assert before and not inside, body
    if kind == "s" and fmt in ("ds", "ss", "oo"):
        # A's columns and x's stored entries merge: only the matched branch knows.
        assert inside and not before, body
    for match in re.finditer(r"wksp_visited = true;", body):
        # Every set left in a loop is in a branch, never at a loop body's top level.
        preceding = body[: match.start()].rstrip().splitlines()
        assert preceding[-1].strip().startswith("wksp +="), preceding[-3:]
        assert preceding[-2].strip().startswith("if ("), preceding[-3:]


@pytest.mark.parametrize("fmt", ["ds", "ss"])
def test_a_dense_result_keeps_its_guard(monkeypatch, fmt):
    """A dense result cell already holds 0: its guard stays, and no flag is declared,
    so a dense result's kernel text is what it was."""
    body = _evaluate_body(_spmv_source(monkeypatch, fmt, "d", "d"))
    assert "if (wksp != 0) {" in body
    assert "wksp_visited" not in body


# --------------------------------------------------------------------------------------
# 2. place_visit_flag on hand-built loops.
# --------------------------------------------------------------------------------------

_FLAG = visit_flag("wksp")
_WKSP = llir.Var(name="wksp", type=llir.DataType.NO_TYPE)


def _int(name: str) -> llir.Var:
    return llir.Var(name=name, type=llir.DataType.INT64)


def _accumulate() -> List[llir.Stmt]:
    return [
        llir.Assign(var=_WKSP, value=_int("v"), op=AssignOp.ADD_ASSIGN),
        visit_flag_set(_FLAG),
    ]


def _loop(body, bound=None, **options) -> llir.ForLoop:
    return llir.ForLoop(
        init=llir.VarInit(var=_int("p"), value=_int("begin")),
        cond=llir.BinOp(op="<", left=_int("p"), right=bound or _int("end")),
        update=llir.Increment(var=_int("p")),
        body=list(body),
        **options,
    )


def _sets(statements) -> int:
    count = 0
    for statement in statements:
        if (
            type(statement) is llir.Assign
            and statement.var == _FLAG
            and statement.value == llir.Literal(True)
        ):
            count += 1
        for name in ("body", "then_body", "else_body"):
            count += _sets(getattr(statement, name, None) or [])
    return count


def test_a_loop_whose_every_iteration_accumulates_tests_its_entry_once():
    entry, loop = place_visit_flag([_loop(_accumulate())], _FLAG)
    assert type(entry) is llir.Assign and entry.var == _FLAG
    assert entry.op is AssignOp.OR_ASSIGN
    assert entry.value == llir.BinOp(op="<", left=_int("begin"), right=_int("end"))
    assert _sets(loop.body) == 0


def test_a_set_in_a_matched_branch_stays_in_it():
    branch = llir.IfThenElse(cond=_int("matched"), then_body=_accumulate())
    (loop,) = place_visit_flag([_loop([branch])], _FLAG)
    assert _sets([loop]) == 1 and loop.body[0].then_body[1] == visit_flag_set(_FLAG)


@pytest.mark.parametrize(
    "body, bound, options",
    [
        ([llir.Continue(), *_accumulate()], None, {}),
        ([llir.Break(), *_accumulate()], None, {}),
        (_accumulate(), llir.FunctionCall(name="end_of", args=[_int("p0")]), {}),
        (_accumulate(), llir.BinOp(op="+", left=_int("p"), right=_int("end")), {}),
        (_accumulate(), None, {"omp_parallel_for": True}),
    ],
    ids=[
        "continue",
        "break",
        "call-bound",
        "bound-reads-the-loop-variable",
        "parallel",
    ],
)
def test_a_loop_the_entry_test_cannot_stand_for_keeps_its_set(body, bound, options):
    (loop,) = place_visit_flag([_loop(body, bound, **options)], _FLAG)
    assert _sets([loop]) == 1


def test_a_while_loop_keeps_its_set():
    (loop,) = place_visit_flag(
        [llir.WhileLoop(cond=_int("going"), body=_accumulate())], _FLAG
    )
    assert _sets([loop]) == 1


def test_nested_loops_test_the_inner_entry_in_the_outer_body():
    inner = _loop(_accumulate())
    outer = llir.ForLoop(
        init=llir.VarInit(var=_int("q"), value=llir.Literal(0)),
        cond=llir.BinOp(op="<", left=_int("q"), right=_int("rows")),
        update=llir.Increment(var=_int("q")),
        body=[inner],
    )
    (placed,) = place_visit_flag([outer], _FLAG)
    entry, loop = placed.body
    assert entry.op is AssignOp.OR_ASSIGN and entry.var == _FLAG
    assert loop is inner and _sets([placed]) == 0


# --------------------------------------------------------------------------------------
# 3. Execution, both compilers.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
@pytest.mark.parametrize("kind", ["d", "s"])
@pytest.mark.parametrize("fmt", _FORMATS)
def test_spmv_keeps_every_visited_row(monkeypatch, fmt, kind, dtype, arm):
    _set_arm(monkeypatch, arm)
    matrix, cells = _matrix(fmt, dtype)
    vector, stored_x = _vector(kind, dtype)
    result = ops.spmv(matrix, vector, output_format="s")
    _assert_result(result, "s", cells, stored_x, dtype)


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("fmt", _FORMATS)
def test_spmv_into_a_coordinate_result_keeps_every_visited_row(monkeypatch, fmt, arm):
    _set_arm(monkeypatch, arm)
    matrix, cells = _matrix(fmt, torch.float64)
    vector, stored_x = _vector("d", torch.float64)
    result = ops.spmv(matrix, vector, output_format="o")
    _assert_result(result, "o", cells, stored_x, torch.float64)


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("fmt", ["ss", "oo", "ds"])
def test_matmul_into_a_sparse_vector_keeps_every_visited_row(monkeypatch, fmt, arm):
    """mlsys-submission's cases through the public product."""
    _set_arm(monkeypatch, arm)
    matrix, cells = _matrix(fmt, torch.float64)
    vector, stored_x = _vector("d", torch.float64)
    result = scorch.matmul(matrix, vector, format="s")
    _assert_result(result, "s", cells, stored_x, torch.float64)


def _spmv_cin(matrix: STensor, vector: STensor, output: str):
    """ops.spmv's index notation: ``y(i) = w`` where ``w += A(i, j) * x(j)``."""

    y = TensorVar("y", shape=(ROWS,), fmt=output, dtype=matrix.dtype)
    a = TensorVar("A", shape=matrix.shape, fmt=matrix.format, dtype=matrix.dtype)
    x = TensorVar("x", shape=vector.shape, fmt=vector.format, dtype=vector.dtype)
    i, j = IndexVar("i"), IndexVar("j")
    workspace = Workspace(name="wksp", dim=0, dtype=y.dtype)
    return ForAll(
        i,
        Where(
            producer=ForAll(
                j,
                TensorAssign(
                    workspace.get_default_access(), a[i, j] * x[j], op=Operation.ADD
                ),
            ),
            consumer=TensorAssign(y[i], workspace.get_default_access()),
        ),
    )


# The new compiler's outcome for _spmv_cin's program: the refusal code it raises, or
# "correct" once it computes the program (test_where_scoped_workspace_reduction.py pins
# the same programs the same way).
_NEW_COMPILER_EXPECTED = "unsupported_statement"


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("kind", ["d", "s"])
@pytest.mark.parametrize("fmt", ["ds", "ss", "oo"])
def test_the_new_compiler_keeps_every_visited_row_or_refuses(fmt, kind, arm):
    """The new compiler on the same program: it keeps the visited rows, or refuses
    the program by the code pinned in _NEW_COMPILER_EXPECTED before any kernel runs.

    auto_options requests a schedule, so the refusal comes before the scheduler:
    lower_cin.refuse_where_program raises the lowering's own LoopIRLoweringError
    unsupported_statement ("Where/workspace programs are outside the dense LoopIR
    families"), the code lower_cin._collect_loop_nest raises on the route without a
    schedule.  Without that check the scheduler planned the loops above the Where alone,
    and its plan verification refused the plan with no code.  On the route without a
    schedule (test_where_scoped_workspace_reduction.py) 0060 also fixes the two refusals
    that came before the lowering's: full CIN verification's free_reduction_conflict
    (the reduced-index rule ignored that the scalar workspace is declared inside
    ForAll(i), so i counted as reduced while free in y[i]; sapling 82883 and 82947), and
    the runtime binding's tensor_shape_rank_mismatch (it gave the workspace the result's
    shape).  Here neither runs: both come after the refusal.  A new compiler that lowers
    a Where computes the program: _NEW_COMPILER_EXPECTED becomes "correct", and this
    test then checks the rows it keeps.
    """
    matrix, cells = _matrix(fmt, torch.float64)
    vector, stored_x = _vector(kind, torch.float64)
    try:
        result = execute_cin_via_loopir(
            _spmv_cin(matrix, vector, "s"),
            (ROWS,),
            matrix,
            vector,
            compile_options=auto_options(bool(arm), jit=True),
        )
    except LoopIRLoweringError as error:
        assert error.defect.code == _NEW_COMPILER_EXPECTED, error
        return
    if isinstance(result, tuple):
        result = result[0]
    assert (
        _NEW_COMPILER_EXPECTED == "correct"
    ), "the new compiler ran a program it refuses"
    _assert_result(result, "s", cells, stored_x, torch.float64)
