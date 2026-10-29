"""A sparse workspace drained into a result whose levels are all dense is computed by
the shipped compiler, in both automatic arms, by a scatter into the result.

The shipped lowering drains a sparse workspace by assembling a compressed or
coordinate level: it sorts the entries and appends one coordinate and one value per
key.  A result whose levels are all dense has neither.  Its values are one array of
its full size, zeroed before the loop nest runs, and a cell's position is arithmetic.
The kernel the drain printed for such a result called ``size``, ``resize`` and
``scorch_vector_set`` on that array, and its C++ did not build (``dd = oo dd`` with a
Gustavson workspace); series step 16b then refused it with
``unsupported_workspace_result_level``.  It is now lowered as a scatter
(``CINLowerer._lower_dense_result_scatter``): each entry's sum is stored into the cell
its key and the enclosing loops name, in insertion order, with no sort for a key of
one variable and no coordinate appended.  Under a reduction loop around the Where the
partial sums are added into their cells, so that program, refused before with
``unsupported_workspace_inside_reduction``, is computed too.  A workspace keyed on one
variable is made once per worker, before the parallel region (once before the
outermost loop of a serial kernel), with capacity the extent of the result level its
key stands for, and cleared after each drain; it was made and freed in every
iteration of the loop around the Where.

The programs: ``C(i, j) = w(j)`` where ``w(j) += A(i, k) * B(k, j)``, with C stored
``dd``, A ``ds``, ``ss`` (row 1 empty) or ``oo`` and B ``ds`` or ``dd``; the result
transposed, so that the key is its first level and the loop binds its second; the
Where inside the reduction loop over ``k``; a workspace keyed on two variables, in
the result's level order and swapped; and a key at the middle level of a rank-3
result, below one loop and above another.  Each is compared with a float64 reference
to a relative 1e-12, and the plain one is also equal, bit for bit, to the same program
into an ``sd`` result, whose sorted drain appends the sums the same producer left in
the same workspace.  (A dense workspace is not bit-for-bit comparable: its
``w[j] += a * b`` is compiled to a fused multiply-add, rounded once, while a sparse
workspace receives the rounded product and adds it.)  Larger and
wider operands give the parallel loop over rows several workers, each reusing its
workspace for many rows, and keys past 1024 coordinates.

The controls are unchanged: a sparse workspace drained into an ``sd`` result is
still assembled by the sorted drain, and a Where below no loop is still computed
through a coordinate intermediate and a conversion.  The new compiler (LoopIR) lowers
no Where program and refuses every one with its general refusal,
``unsupported_statement``; it gave this drain the shipped lowering's refusal code only
while the shipped lowering refused it too.
"""

from __future__ import annotations

from typing import Any, Tuple

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
from scorch.compiler.loopir.lower_cin import (  # type: ignore[import-untyped]
    LoopIRLoweringError,
)
from scorch.compiler.loopir.pipeline import (  # type: ignore[import-untyped]
    execute_cin_via_loopir,
)
from scorch import utils  # type: ignore[import-untyped]
from scorch.utils import (  # type: ignore[import-untyped]
    _kernel_name,
    jit_preamble_text,
    packaged_header_text,
)
from tests.test_scorch.test_helpers import lower_and_print
from tests.test_scorch.test_sparse_assembly_strategy import auto_options

F64 = torch.float64
N, K, M, L = 7, 6, 5, 4


def _set_arm(monkeypatch: pytest.MonkeyPatch, arm: int) -> None:
    monkeypatch.setenv("SCORCH_REGBLOCK", str(arm))
    assert CompileOptions.from_environment().scheduler.regblock_enabled == bool(arm)


def _random(*shape: int, keep: float, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(*shape, generator=generator, dtype=F64)
    values[values < 1.0 - keep] = 0.0
    return values


def _wrap(values: torch.Tensor, fmt: str) -> STensor:
    tensor = STensor.from_torch(values.clone())
    return tensor.to_dense() if set(fmt) == {"d"} else tensor.to_sparse(fmt)


def _operands(
    a_format: str, b_format: str = "ds", n: int = N, k: int = K, m: int = M
) -> Tuple[STensor, STensor, torch.Tensor]:
    a = _random(n, k, keep=0.5, seed=16)
    a[1] = 0.0
    b = _random(k, m, keep=0.6, seed=17)
    return _wrap(a, a_format), _wrap(b, b_format), a @ b


def _program(
    a_format: str,
    result_format: str,
    dense_workspace: bool,
    b_format: str = "ds",
    shape: Tuple[int, int, int] = (N, K, M),
) -> Any:
    n, k_extent, m = shape
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt=result_format, shape=(n, m), dtype=F64)
    a = TensorVar("A", fmt=a_format, shape=(n, k_extent), dtype=F64)
    b = TensorVar("B", fmt=b_format, shape=(k_extent, m), dtype=F64)
    w = Workspace(name="w", dim=1, dtype=F64, dense=dense_workspace)
    return ForAll(
        i,
        Where(
            producer=ForAll(
                k, ForAll(j, TensorAssign(w[j], a[i, k] * b[k, j], op=Operation.ADD))
            ),
            consumer=ForAll(j, TensorAssign(c[i, j], w[j])),
        ),
    )


def _program_transposed(shape: Tuple[int, int, int] = (N, K, M)) -> Any:
    n, k_extent, m = shape
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dd", shape=(m, n), dtype=F64)
    a = TensorVar("A", fmt="ds", shape=(n, k_extent), dtype=F64)
    b = TensorVar("B", fmt="ds", shape=(k_extent, m), dtype=F64)
    w = Workspace(name="w", dim=1, dtype=F64, dense=False)
    return ForAll(
        i,
        Where(
            producer=ForAll(
                k, ForAll(j, TensorAssign(w[j], a[i, k] * b[k, j], op=Operation.ADD))
            ),
            consumer=ForAll(j, TensorAssign(c[j, i], w[j])),
        ),
    )


def _program_below_no_loop() -> Any:
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dd", shape=(N, M), dtype=F64)
    a = TensorVar("A", fmt="ds", shape=(N, K), dtype=F64)
    b = TensorVar("B", fmt="ds", shape=(K, M), dtype=F64)
    w = Workspace(name="w", dim=2, dtype=F64, dense=False)
    return Where(
        producer=ForAll(
            i,
            ForAll(
                k,
                ForAll(j, TensorAssign(w[i, j], a[i, k] * b[k, j], op=Operation.ADD)),
            ),
        ),
        consumer=ForAll(i, ForAll(j, TensorAssign(c[i, j], w[i, j]))),
    )


def _program_inside_a_reduction(shape: Tuple[int, int, int] = (N, K, M)) -> Any:
    n, k_extent, m = shape
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    c = TensorVar("C", fmt="dd", shape=(n, m), dtype=F64)
    a = TensorVar("A", fmt="ds", shape=(n, k_extent), dtype=F64)
    b = TensorVar("B", fmt="ds", shape=(k_extent, m), dtype=F64)
    w = Workspace(name="w", dim=1, dtype=F64, dense=False)
    return ForAll(
        i,
        ForAll(
            k,
            Where(
                producer=ForAll(
                    j, TensorAssign(w[j], a[i, k] * b[k, j], op=Operation.ADD)
                ),
                consumer=ForAll(j, TensorAssign(c[i, j], w[j], op=Operation.ADD)),
            ),
        ),
    )


def _program_two_key(swapped: bool) -> Any:
    """C(i, j, l) = sum_k A(i, k) B(k, j, l) through a workspace keyed (j, l), or
    (l, j) when ``swapped``: the key's storage order is then not the result's."""

    i, j, k, l_ = IndexVar("i"), IndexVar("j"), IndexVar("k"), IndexVar("l")
    c = TensorVar("C", fmt="ddd", shape=(N, M, L), dtype=F64)
    a = TensorVar("A", fmt="ds", shape=(N, K), dtype=F64)
    b = TensorVar("B", fmt="ddd", shape=(K, M, L), dtype=F64)
    w = Workspace(name="w", dim=2, dtype=F64, dense=False)
    key = (l_, j) if swapped else (j, l_)
    product = TensorAssign(w[key], a[i, k] * b[k, j, l_], op=Operation.ADD)
    return ForAll(
        i,
        Where(
            producer=ForAll(k, ForAll(j, ForAll(l_, product))),
            consumer=ForAll(j, ForAll(l_, TensorAssign(c[i, j, l_], w[key]))),
        ),
    )


def _program_middle_key() -> Any:
    """C(a, b, c) = sum_d A(a, d) B(d, b, c) with loops over a and c around a
    workspace over b, the result's middle level."""

    va, vb, vc, vd = IndexVar("a"), IndexVar("b"), IndexVar("c"), IndexVar("d")
    c = TensorVar("C", fmt="ddd", shape=(N, M, L), dtype=F64)
    a = TensorVar("A", fmt="ds", shape=(N, K), dtype=F64)
    b = TensorVar("B", fmt="ddd", shape=(K, M, L), dtype=F64)
    w = Workspace(name="w", dim=1, dtype=F64, dense=False)
    product = TensorAssign(w[vb], a[va, vd] * b[vd, vb, vc], op=Operation.ADD)
    return ForAll(
        va,
        ForAll(
            vc,
            Where(
                producer=ForAll(vd, ForAll(vb, product)),
                consumer=ForAll(vb, TensorAssign(c[va, vb, vc], w[vb])),
            ),
        ),
    )


def _rank3_operands() -> Tuple[STensor, STensor, torch.Tensor, torch.Tensor]:
    a = _random(N, K, keep=0.5, seed=16)
    a[1] = 0.0
    b = _random(K, M, L, keep=0.6, seed=18)
    return _wrap(a, "ds"), _wrap(b, "ddd"), a, b


def _dense(result: Any) -> torch.Tensor:
    values = result.to_torch()
    return values if values.layout == torch.strided else values.to_dense()


def _close(got: Any, want: torch.Tensor) -> None:
    values = _dense(got)
    assert values.shape == want.shape, (values.shape, want.shape)
    assert torch.allclose(values, want, rtol=1e-12, atol=0.0)


def _new_compiler_refusal(program: Any, a: STensor, b: STensor, arm: int) -> Any:
    with pytest.raises(LoopIRLoweringError) as caught:
        execute_cin_via_loopir(
            program, (N, M), a, b, compile_options=auto_options(bool(arm), jit=True)
        )
    return caught.value


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("a_format", ["ds", "ss", "oo"])
def test_the_shipped_compiler_computes_the_drain(
    monkeypatch: pytest.MonkeyPatch, a_format: str, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    a, b, want = _operands(a_format)
    _close(ops.lower_and_exec_cin(_program(a_format, "dd", False), (N, M), a, b), want)


@pytest.mark.parametrize("arm", [0, 1])
def test_dd_oo_dd_through_a_gustavson_workspace(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    """The program whose kernel did not build: a COO operand times a dense one."""

    _set_arm(monkeypatch, arm)
    a, b, want = _operands("oo", "dd")
    program = _program("oo", "dd", False, b_format="dd")
    _close(ops.lower_and_exec_cin(program, (N, M), a, b), want)


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("a_format", ["ds", "ss"])
def test_the_scatter_stores_what_the_sorted_drain_stores(
    monkeypatch: pytest.MonkeyPatch, a_format: str, arm: int
) -> None:
    """The same producer fills the same workspace for a ``dd`` and an ``sd`` result,
    so the scatter's cells hold, bit for bit, the values the sorted drain appends."""

    _set_arm(monkeypatch, arm)
    a, b, _ = _operands(a_format)
    scattered = ops.lower_and_exec_cin(_program(a_format, "dd", False), (N, M), a, b)
    drained = ops.lower_and_exec_cin(_program(a_format, "sd", False), (N, M), a, b)
    assert torch.equal(_dense(scattered), _dense(drained))


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize(
    "shape", [(200, 150, 170), (40, 30, 2500)], ids=["parallel", "wide"]
)
def test_larger_operands(
    monkeypatch: pytest.MonkeyPatch, shape: Tuple[int, int, int], arm: int
) -> None:
    """Enough rows for the parallel loop to have several workers, each draining its
    one workspace for many rows, so a ``clear`` that left an entry behind would add it
    into the next row's cells; and keys past 1024 coordinates, the capacity the
    workspace had when it was made in every iteration.

    The parallel loop is the loop over rows, and the kernel asks for
    ``scorch_nthreads(A's entries, rows)`` = min(rows / 16, entries / 500, the thread
    ceiling) workers: min(12, about 30) at 200 rows.  Under the reduction loop each
    worker adds the partial sums of its own rows; a cell's position is computed from
    its row, so no two workers add into one cell."""

    _set_arm(monkeypatch, arm)
    n, k, m = shape
    a, b, want = _operands("ds", n=n, k=k, m=m)
    program = _program("ds", "dd", False, shape=shape)
    _close(ops.lower_and_exec_cin(program, (n, m), a, b), want)
    transposed = ops.lower_and_exec_cin(_program_transposed(shape), (m, n), a, b)
    _close(transposed, want.T)
    reduced = ops.lower_and_exec_cin(_program_inside_a_reduction(shape), (n, m), a, b)
    _close(reduced, want)


@pytest.mark.parametrize("arm", [0, 1])
def test_a_transposed_result(monkeypatch: pytest.MonkeyPatch, arm: int) -> None:
    """The key is the result's first level and the loop binds its second."""

    _set_arm(monkeypatch, arm)
    a, b, want = _operands("ds")
    _close(ops.lower_and_exec_cin(_program_transposed(), (M, N), a, b), want.T)


@pytest.mark.parametrize("arm", [0, 1])
def test_a_drain_inside_a_reduction_adds_its_partial_sums(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    a, b, want = _operands("ds")
    _close(ops.lower_and_exec_cin(_program_inside_a_reduction(), (N, M), a, b), want)


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("swapped", [False, True], ids=["level-order", "swapped"])
def test_a_key_of_two_variables(
    monkeypatch: pytest.MonkeyPatch, swapped: bool, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    a, b, a_values, b_values = _rank3_operands()
    got = ops.lower_and_exec_cin(_program_two_key(swapped), (N, M, L), a, b)
    _close(got, torch.einsum("ik,kjl->ijl", a_values, b_values))


@pytest.mark.parametrize("arm", [0, 1])
def test_a_key_at_the_middle_level(monkeypatch: pytest.MonkeyPatch, arm: int) -> None:
    _set_arm(monkeypatch, arm)
    a, b, a_values, b_values = _rank3_operands()
    got = ops.lower_and_exec_cin(_program_middle_key(), (N, M, L), a, b)
    _close(got, torch.einsum("ad,dbc->abc", a_values, b_values))


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("a_format", ["ds", "ss"])
def test_the_new_compiler_keeps_the_general_refusal(a_format: str, arm: int) -> None:
    a, b, _ = _operands(a_format)
    error = _new_compiler_refusal(_program(a_format, "dd", False), a, b, arm)
    assert error.defect.code == "unsupported_statement", error


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("a_format", ["ds", "ss"])
def test_the_program_with_a_dense_workspace_is_computed(
    monkeypatch: pytest.MonkeyPatch, a_format: str, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    a, b, want = _operands(a_format)
    _close(ops.lower_and_exec_cin(_program(a_format, "dd", True), (N, M), a, b), want)


@pytest.mark.parametrize("arm", [0, 1])
def test_a_sparse_workspace_into_an_sd_result_is_still_computed(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    a, b, want = _operands("ds")
    _close(ops.lower_and_exec_cin(_program("ds", "sd", False), (N, M), a, b), want)


@pytest.mark.parametrize("arm", [0, 1])
def test_the_shipped_compiler_computes_a_where_below_no_loop(
    monkeypatch: pytest.MonkeyPatch, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    a, b, want = _operands("ds")
    _close(ops.lower_and_exec_cin(_program_below_no_loop(), (N, M), a, b), want)


@pytest.mark.parametrize("arm", [0, 1])
def test_the_new_compiler_keeps_the_general_refusal_below_no_loop(arm: int) -> None:
    a, b, _ = _operands("ds")
    error = _new_compiler_refusal(_program_below_no_loop(), a, b, arm)
    assert error.defect.code == "unsupported_statement", error


@pytest.mark.parametrize("arm", [0, 1])
def test_the_new_compiler_keeps_the_general_refusal_inside_a_reduction(
    arm: int,
) -> None:
    a, b, _ = _operands("ds")
    error = _new_compiler_refusal(_program_inside_a_reduction(), a, b, arm)
    assert error.defect.code == "unsupported_statement", error


_PARALLEL_VIEW = "[(size_t)omp_get_thread_num()].make_region_view()"


@pytest.mark.parametrize(
    "program, extent, borrow",
    [
        (lambda: _program("ds", "dd", False), "C1_size", _PARALLEL_VIEW),
        (
            lambda: _program("oo", "dd", False, b_format="dd"),
            "C1_size",
            "[0].make_view()",
        ),
        (_program_transposed, "C0_size", _PARALLEL_VIEW),
        (_program_inside_a_reduction, "C1_size", _PARALLEL_VIEW),
        (_program_middle_key, "C1_size", _PARALLEL_VIEW),
    ],
    ids=[
        "parallel",
        "serial-coo-operand",
        "transposed",
        "inside-a-reduction",
        "middle-key",
    ],
)
def test_the_workspace_is_made_once_per_worker_and_cleared_after_each_drain(
    program: Any, extent: str, borrow: str
) -> None:
    """The scatter's workspace: one ``coo_workspace_owner_1d`` per worker in a pool
    built before the loops, of capacity the extent of the result level its key
    stands for, and each worker's ``coo_workspace_view_1d`` from it, a local of the
    worker, inserted into unchecked and cleared after each drain.  None is made
    inside the loops.  In a parallel region the owners allocate nothing before it
    (``emplace_back(<extent>, true)``): each worker's view is
    ``make_region_view()``, the row body opens with the skip of a failed view, and
    the pool is checked after the region.  A serial kernel (COO ``A``'s loop over
    row segments is not one the parallel marking can split) allocates its one owner
    before the loop and takes ``make_view()``, with no skip and no check.  The
    pool's definitions (``workspace_pool.h``) open the kernel's own source, ahead
    of the function; the checks below read the function."""

    cpp = lower_and_print(program())
    pool_definitions = packaged_header_text("workspace_pool.h").rstrip("\n") + "\n\n"
    assert cpp.startswith(pool_definitions)
    cpp = cpp[len(pool_definitions) :]
    assert "class coo_workspace_owner_1d" not in cpp
    parallel = borrow == _PARALLEL_VIEW
    assert cpp.count("std::vector<coo_workspace_owner_1d<double>> w_pool_owner;") == 1
    in_region = ", true" if parallel else ""
    assert cpp.count(f"w_pool_owner.emplace_back({extent}{in_region});") == 1
    assert cpp.count(f"auto w = w_pool_owner{borrow};") == 1
    assert "coo_workspace_1d<" not in cpp
    assert cpp.count("w.insert_unchecked({") >= 1
    assert "w.insert({" not in cpp
    scatter = cpp[cpp.index("Scatter the sparse workspace") :]
    assert scatter.count("w.clear();") == 1
    assert cpp.count("w.clear();") == 1
    assert cpp.count("if (w.failed())") == (1 if parallel else 0)
    assert cpp.count("scorch_check_workspace_pool(w_pool_owner);") == (
        1 if parallel else 0
    )
    if parallel:
        # The skip opens the row body, inside the region; the check follows the
        # region, in serial code, before the result is assembled.
        assert cpp.index("auto w = w_pool_owner") < cpp.index("if (w.failed())")
        assert cpp.index("w.clear();") < cpp.index("scorch_check_workspace_pool(")
        assert cpp.index("scorch_check_workspace_pool(") < cpp.index(
            "Assemble final result"
        )


@pytest.mark.parametrize(
    "program, holds_a_pool",
    [
        (lambda: _program("ds", "dd", False), True),
        (lambda: _program("oo", "dd", False, b_format="dd"), True),
        (lambda: _program("ds", "dd", True), False),
        (lambda: _program("ds", "sd", False), False),
    ],
    ids=["scatter", "serial-scatter", "dense-workspace", "sorted-drain"],
)
def test_only_a_pool_kernel_carries_the_pool_definitions_and_its_name_covers_them(
    monkeypatch: pytest.MonkeyPatch, program: Any, holds_a_pool: bool
) -> None:
    """The pool's definitions are not in the preamble every kernel shares: code
    generation puts ``workspace_pool.h`` in the source of a kernel that declares a
    pool, and of no other.  A kernel without one (a dense workspace, or a sparse
    workspace drained in sorted order into an ``sd`` result) is emitted as before,
    and an edit to the header leaves its source unchanged.  A pool
    kernel's name is made from its source, so a change to the header renames it and
    its cached build is not reused; any other kernel keeps its name."""

    preamble = jit_preamble_text()
    assert "coo_workspace_owner_1d" not in preamble
    pool_text = packaged_header_text("workspace_pool.h")
    cpp = lower_and_print(program())
    assert cpp.startswith(pool_text.rstrip("\n") + "\n\n") is holds_a_pool
    assert ("class coo_workspace_owner_1d" in cpp) is holds_a_pool
    assert ("std::vector<coo_workspace_owner_1d" in cpp) is holds_a_pool
    name = _kernel_name(preamble, cpp)

    edited = pool_text + "// edited\n"
    monkeypatch.setattr(utils, "packaged_header_text", lambda filename: edited)
    cpp_after_edit = lower_and_print(program())
    if holds_a_pool:
        function = cpp[len(pool_text.rstrip("\n") + "\n\n") :]
        assert cpp_after_edit == edited.rstrip("\n") + "\n\n" + function
        assert _kernel_name(preamble, cpp_after_edit) != name
    else:
        assert cpp_after_edit == cpp
        assert _kernel_name(preamble, cpp_after_edit) == name


def test_the_controls_keep_a_workspace_made_in_every_iteration() -> None:
    """Kernels this change does not touch: the sorted drain into an ``sd`` result
    (the shipped kernel), and a key of two variables, whose ``coo_workspace<T, 2>`` is
    a hash map with no ``clear`` and no flag array as long as an extent."""

    sorted_drain = lower_and_print(_program("ds", "sd", False))
    assert "auto w = coo_workspace_1d<double, 1>(1024);" in sorted_drain
    assert "w_pool_owner" not in sorted_drain
    two_key = lower_and_print(_program_two_key(False))
    assert "coo_workspace<double, 2> w = coo_workspace<double, 2>(1024" in two_key
    assert "w_pool_owner" not in two_key
