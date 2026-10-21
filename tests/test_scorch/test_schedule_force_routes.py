"""A forced schedule reaches every route ``scorch.matmul`` has.

``schedule_force(S)`` used to be ignored wherever a product was served by a prebuilt kernel
or a plan, none of which takes a schedule: the C entry point's five routes (an STensor's
plan, a torch CSR operand's plan, its unplanned drop-in SpMM, the gradient-recording plan,
``x @ A``'s transposed plan), the Python function's own plan probe, and autograd.py's two
routes (``x @ A`` computed unrecorded, and a recorded product's forward). Each returned the
right values under the default schedule, silently (sapling 84220, 84232). Now every one of
them goes to the generated kernel, which honours the schedule or refuses it with a code.

Each test records the generated route's calls (``ops._generated_matmul``) with the schedule
forced at the moment of the call, so "the schedule's kernel ran" is read directly rather
than inferred from a compile count, and checks that no plan or prebuilt route counted a
product. A recorded product's backward runs under the default schedule even when called
inside the context: a schedule names the forward expression only (``schedule_force``).

The C entry point and the Python readers read the context variable only once schedule_force
has run in the process (scheduler.FORCE_SEEN_CELL and the C entry point's own flag), so the
last four tests check that the flag changes no result: a schedule forced in another thread
leaves this thread's call to its plan; a context copied inside the block keeps the schedule
after the block exits; once a block has run, a call outside every block is still served by its
plan; and a fresh process has neither flag set until its first block sets both, before the
variable.
"""

import asyncio
import contextvars
import subprocess
import sys
import threading
import warnings

import numpy as np
import pytest
import torch

import scorch
import scorch_ops
from scorch import autograd as scorch_autograd
from scorch import ops
from scorch import plan as plan_mod
from scorch.compiler.diagnostics import ScheduleNotThreadedThroughSpMVError
from scorch.compiler.scheduler import (
    FORCE_SEEN_CELL,
    call_without_forced_schedule,
    get_forced_schedule,
)
from scorch.plan import plans_of, transposed_plans_of

warnings.filterwarnings("ignore", message=".*Sparse.*")

pytestmark = pytest.mark.skipif(
    not hasattr(scorch_ops, "set_planned_matmul_schedule_force"),
    reason="this build's C entry point does not read the forced schedule",
)

# test_schedule_api's strip-mined SpMM: its generated C++ differs from every default kernel.
S = scorch.Schedule(
    loop_order=("i", "j", "k"),
    tiles=(scorch.TileSpec("i", 4, placement="outermost", accum="direct"),),
    tag="tile-i-4",
)
N, K = 48, 8


@pytest.fixture(autouse=True)
def plans_on():
    enabled = plan_mod.enabled()
    plan_mod.set_enabled(True)
    yield
    plan_mod.set_enabled(enabled)


@pytest.fixture
def generated(monkeypatch):
    """The schedule forced at each call of the generated route, in order."""
    seen = []
    real = ops._generated_matmul

    def spy(*args, **kwargs):
        seen.append(get_forced_schedule())
        return real(*args, **kwargs)

    monkeypatch.setattr(ops, "_generated_matmul", spy)
    return seen


def dense_a(seed=0):
    rng = np.random.default_rng(seed)
    a = np.where(rng.random((N, N)) < 0.1, rng.uniform(-1, 1, (N, N)), 0.0)
    a[np.arange(N), (np.arange(N) * 7) % N] = 0.5  # no empty row
    return a


def torch_csr(seed=0):
    return torch.from_numpy(dense_a(seed)).to_sparse_csr()


def stensor(seed=0):
    return scorch.from_torch(torch.from_numpy(dense_a(seed)), "A").to_sparse("ds")


def rhs(rows, cols, seed=1):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.uniform(-1, 1, (rows, cols)))


def close(got, want):
    got = got.detach().to_dense() if got.layout != torch.strided else got.detach()
    want = np.asarray(want)
    assert np.abs(got.numpy() - want).max() <= 1e-12 * max(1.0, np.abs(want).max())


def served(x):
    return sum(p.served for p in plans_of(x).values())


def test_stensor_plan(generated):
    a, d = stensor(), rhs(N, K)
    ops.matmul(a, d), ops.matmul(a, d)
    assert plans_of(a), "no plan was installed"
    before = served(a)
    with scorch.schedule_force(S):
        out = ops.matmul(a, d)
    close(out, dense_a() @ d.numpy())
    assert generated == [S]
    assert served(a) == before


def test_torch_csr_plan(generated):
    a, d = torch_csr(), rhs(N, K)
    ops.matmul(a, d), ops.matmul(a, d)
    assert plans_of(a), "no plan was installed"
    before = served(a)
    with scorch.schedule_force(S):
        out = ops.matmul(a, d)
    close(out, dense_a() @ d.numpy())
    assert generated == [S]
    assert served(a) == before


def test_torch_csr_unplanned_drop_in(generated):
    a, d = torch_csr(), rhs(N, K)
    before = scorch_ops.planned_matmul_ordinary_served()
    with scorch.schedule_force(S):
        out = ops.matmul(a, d)
    close(out, dense_a() @ d.numpy())
    assert generated == [S]
    assert scorch_ops.planned_matmul_ordinary_served() == before


def test_python_plan_probe_alone(generated):
    """The Python function's own probe, with no C entry point in the call."""
    a, d = stensor(), rhs(N, K)
    ops._matmul_python(a, d), ops._matmul_python(a, d)
    assert plans_of(a), "no plan was installed"
    before = served(a)
    with scorch.schedule_force(S):
        out = ops._matmul_python(a, d)
    close(out, dense_a() @ d.numpy())
    assert generated == [S]
    assert served(a) == before


def test_transposed_plan(generated):
    a, x = torch_csr(), rhs(K, N)
    for _ in range(3):
        ops.matmul(x, a)
    assert transposed_plans_of(a), "no transposed plan was installed"
    with scorch.schedule_force(S):
        out = ops.matmul(x, a)
    close(out, x.numpy() @ dense_a())
    assert generated == [S]


def test_dense_times_sparse_unrecorded(generated):
    """autograd.right_product, which computes x @ A on the drop-in kernel when no gradient
    is wanted, leaves it to the generated kernel under a forced schedule."""
    a, x = torch_csr(), rhs(K, N)
    with scorch.schedule_force(S):
        out = ops.matmul(x, a)
    close(out, x.numpy() @ dense_a())
    assert generated == [S]


def gradients(a, other, left):
    """(A.grad as dense, other.grad) of sum(a @ other) or sum(other @ a)."""
    out = ops.matmul(a, other) if left else ops.matmul(other, a)
    out.sum().backward()
    return a.grad.to_dense().clone(), other.grad.clone(), out.detach()


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("planned", [False, True])
def test_recorded_forward_honours_the_schedule_and_backward_does_not(generated, left, planned):
    """A product a gradient is wanted through: its forward is the schedule's kernel, its
    backward the same gradients as without the schedule, computed under the default one --
    with backward() called inside the context, where only the forward may see S."""
    other = rhs(N, K) if left else rhs(K, N)
    a = torch_csr()
    if planned:
        for _ in range(3):
            ops.matmul(a, other) if left else ops.matmul(other, a)
    a_ref, o_ref = a.clone().requires_grad_(True), other.clone().requires_grad_(True)
    want_a, want_o, want_out = gradients(a_ref, o_ref, left)
    a.requires_grad_(True)
    other = other.clone().requires_grad_(True)
    del generated[:]
    with scorch.schedule_force(S):
        out = ops.matmul(a, other) if left else ops.matmul(other, a)
        forward_calls = list(generated)
        out.sum().backward()
    assert forward_calls == [S]
    assert all(s is None for s in generated[len(forward_calls):])
    close(out, want_out.numpy())
    close(a.grad.to_dense(), want_a.numpy())
    close(other.grad, want_o.numpy())


def test_the_einsums_derivative_runs_under_the_default_schedule(
    generated, monkeypatch
):
    """A product autograd.py's Functions do not cover (use_cache=False asks for the
    generated kernel) is recorded with the einsum's derivative (einsum_grad), which
    computes its gradients with scorch.einsum.  Those are other expressions than the
    forward the schedule names, so, like autograd.py's own products, they run under the
    default schedule with backward() called inside the context; the forward is the
    schedule's kernel."""
    seen = []
    real = ops.einsum

    def spy(*args, **kwargs):
        seen.append(get_forced_schedule())
        return real(*args, **kwargs)

    monkeypatch.setattr(ops, "einsum", spy)
    a_ref = torch_csr().requires_grad_(True)
    o_ref = rhs(N, K).requires_grad_(True)
    want_out = ops.matmul(a_ref, o_ref, use_cache=False)
    want_out.sum().backward()
    a = torch_csr().requires_grad_(True)
    other = rhs(N, K).requires_grad_(True)
    del generated[:], seen[:]
    with scorch.schedule_force(S):
        out = ops.matmul(a, other, use_cache=False)
        forward_calls, before = list(generated), len(seen)
        out.sum().backward()
    assert forward_calls == [S]
    assert all(s is None for s in generated[len(forward_calls):])
    assert len(seen) > before, "the derivative computed nothing with scorch.einsum"
    assert all(s is None for s in seen[before:])
    close(out, want_out.detach().numpy())
    close(a.grad.to_dense(), a_ref.grad.to_dense().numpy())
    close(other.grad, o_ref.grad.numpy())


@pytest.mark.parametrize(
    "call",
    ["torch_first", "torch_cached", "stensor_direct", "recorded"],
)
def test_matrix_vector_refuses_with_its_code(call):
    """No route takes a schedule for a matrix times a vector, so every one refuses it."""
    v = torch.from_numpy(np.random.default_rng(2).uniform(-1, 1, N))
    if call == "stensor_direct":
        # spmv takes STensor operands (series step 2); the vector is one.
        a, v = stensor(), scorch.from_torch(v, "v")
        ops.spmv(a, v), ops.spmv(a, v)
        with scorch.schedule_force(S), pytest.raises(ScheduleNotThreadedThroughSpMVError):
            ops.spmv(a, v)
        return
    a = torch_csr()
    if call == "torch_cached":
        ops.matmul(a, v), ops.matmul(a, v)
    if call == "recorded":
        a.requires_grad_(True)
    with scorch.schedule_force(S), pytest.raises(ScheduleNotThreadedThroughSpMVError):
        ops.matmul(a, v)


def test_call_without_forced_schedule():
    def inner():
        return get_forced_schedule()

    assert call_without_forced_schedule(inner) is None
    with scorch.schedule_force(S):
        assert call_without_forced_schedule(inner) is None
        assert get_forced_schedule() is S

        def boom():
            raise KeyError("x")

        with pytest.raises(KeyError):
            call_without_forced_schedule(boom)
        assert get_forced_schedule() is S
    assert get_forced_schedule() is None


def planned_torch_csr(d):
    a = torch_csr()
    ops.matmul(a, d), ops.matmul(a, d)
    assert plans_of(a), "no plan was installed"
    return a


def test_another_threads_schedule_leaves_this_threads_call_to_its_plan(generated):
    """A schedule forced in another thread is that thread's: a call made here while the
    other thread's block is open is served by its plan, with the plan's result."""
    d = rhs(N, K)
    a = planned_torch_csr(d)
    entered, release = threading.Event(), threading.Event()

    def hold():
        with scorch.schedule_force(S):
            entered.set()
            release.wait(60)

    other = threading.Thread(target=hold)
    other.start()
    try:
        assert entered.wait(60)
        assert FORCE_SEEN_CELL[0] and scorch_ops.planned_matmul_schedule_force_seen()
        before = served(a)
        out = ops.matmul(a, d)
    finally:
        release.set()
        other.join(60)
    close(out, dense_a() @ d.numpy())
    assert generated == []
    assert served(a) == before + 1


@pytest.mark.parametrize("carrier", ["copy_context", "asyncio_task"])
def test_a_context_copied_inside_the_block_keeps_the_schedule(generated, carrier):
    """A context copied inside schedule_force keeps the schedule after the block exits: a
    call run in it is the schedule's kernel's, for a planned operand and an unplanned one,
    and neither a plan nor the C entry point's drop-in kernel counts a product."""
    d = rhs(N, K)
    a = planned_torch_csr(d)
    fresh = torch_csr(seed=3)

    def both():
        return ops.matmul(a, d), ops.matmul(fresh, d)

    if carrier == "copy_context":
        with scorch.schedule_force(S):
            copied = contextvars.copy_context()
        assert get_forced_schedule() is None
        before = served(a), scorch_ops.planned_matmul_ordinary_served()
        out, out_fresh = copied.run(both)
    else:

        async def later():
            await asyncio.sleep(0)
            return both()

        async def main():
            with scorch.schedule_force(S):
                task = asyncio.ensure_future(later())
            assert get_forced_schedule() is None
            return await task

        before = served(a), scorch_ops.planned_matmul_ordinary_served()
        out, out_fresh = asyncio.run(main())
    close(out, dense_a() @ d.numpy())
    close(out_fresh, dense_a(3) @ d.numpy())
    assert generated == [S, S]
    assert (served(a), scorch_ops.planned_matmul_ordinary_served()) == before


def test_after_a_block_a_call_outside_every_block_is_served_by_its_plan(generated):
    """The flags only enable the read of the context variable: once a block has run, both
    are set, and a call outside every block is served by its plan (C entry point) or by the
    Python function's plan probe, with the plan's result."""
    d = rhs(N, K)
    a = planned_torch_csr(d)
    st = stensor()
    ops._matmul_python(st, d), ops._matmul_python(st, d)
    assert plans_of(st), "no plan was installed"
    with scorch.schedule_force(S):
        pass
    assert FORCE_SEEN_CELL[0] and scorch_ops.planned_matmul_schedule_force_seen()
    before = served(a), served(st)
    out = ops.matmul(a, d)
    out_python = ops._matmul_python(st, d)
    close(out, dense_a() @ d.numpy())
    close(out_python, dense_a() @ d.numpy())
    assert generated == []
    assert (served(a), served(st)) == (before[0] + 1, before[1] + 1)


_FRESH_PROCESS = """
import scorch, scorch_ops
from scorch.compiler import scheduler

def flags():
    return scheduler.FORCE_SEEN_CELL[0], scorch_ops.planned_matmul_schedule_force_seen()

print("fresh", *flags())
seen = []
# After ops.py's hook, which sets the C entry point's flag: this one sees that flag set,
# the Python flag not yet, and the variable not yet holding the schedule.
scheduler._ON_FIRST_SCHEDULE_FORCE.append(
    lambda: seen.append((scheduler.get_forced_schedule(), *flags()))
)
with scorch.schedule_force(None):
    print("after a block without a schedule", *flags())
with scorch.schedule_force(scorch.Schedule(loop_order=("i", "j", "k"))):
    print("inside", *flags())
with scorch.schedule_force(scorch.Schedule(loop_order=("i", "j", "k"))):
    pass
print("hooks", len(seen), *seen[0])
"""


def test_a_fresh_process_sets_both_flags_at_its_first_block_before_the_variable():
    """In a process of its own, since this one's other tests set both flags: neither is set
    by importing scorch or by a block without a schedule; the first block with one sets
    both, the C entry point's before the context variable holds the schedule; the hooks
    run once."""
    run = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    lines = [
        line
        for line in run.stdout.splitlines()
        if line.startswith(("fresh", "after", "inside", "hooks"))
    ]
    assert lines == [
        "fresh False False",
        "after a block without a schedule False False",
        "inside True True",
        "hooks 1 None False True",
    ]


# ---- A forced schedule changes the kernel, never the result (step 10 v3) ----------------------
#
# Every call shape matmul can divert to the generated kernel, each at two extents and with its
# sparse operand in both stored orders (an STensor in mode order [1, 0], a torch CSC tensor), is
# run unforced and then under the strip-mined S. Forced, the result must have the unforced
# result's type, layout, format, mode order, shape and dtype and the same float64 values, or a
# refusal carrying one of the codes in _REFUSALS, each for the reason it names: never another
# format. A product of an STensor operand, or of a torch CSR operand stored in its own order,
# with a dense result must run S's kernel (_MUST_RUN). (A schedule without tiles is refused by
# the generated pipeline, unsupported_requested_schedule, so it is no control here.) A matrix
# times a vector refuses every schedule (test_matrix_vector_refuses_with_its_code).

from scorch.exceptions import defect_code  # noqa: E402
from scorch.stensor import STensor  # noqa: E402

def _sparse(n, kind, transposed_storage, seed=0):
    rng = np.random.default_rng(seed)
    d = np.where(rng.random((n, n)) < 0.1, rng.uniform(-1, 1, (n, n)), 0.0)
    d[np.arange(n), (np.arange(n) * 7) % n] = 0.5
    t = torch.from_numpy(d)
    if kind == "torch_csr":
        return t.to_sparse_csc() if transposed_storage else t.to_sparse_csr()
    if kind == "torch_coo":
        return t.to_sparse_coo().coalesce()
    s = scorch.from_torch(t, "A").to_sparse("ds" if kind == "stensor" else "oo")
    if transposed_storage:
        s = s.copy()
        s.change_mode_order([1, 0])
    return s


def _dense(rows, cols, seed=1):
    return torch.from_numpy(np.random.default_rng(seed).uniform(-1, 1, (rows, cols)))


def _calls(n, order):
    """(name, operands, product) for every divertible call shape: ``operands()`` builds fresh
    operands, outside any forced block (building one runs conversion kernels of its own),
    and ``product(*operands)`` is the call."""
    k = 8
    sp = lambda kind, seed=0: _sparse(n, kind, order, seed)  # noqa: E731
    vec = lambda: torch.from_numpy(np.random.default_rng(3).uniform(-1, 1, n))  # noqa: E731
    mm = ops.matmul
    grad = lambda t: t.requires_grad_(True)  # noqa: E731
    return [
        ("stensor@dense", lambda: (sp("stensor"), _dense(n, k)), mm),
        ("torch_csr@dense", lambda: (sp("torch_csr"), _dense(n, k)), mm),
        ("dense@torch_csr", lambda: (_dense(k, n), sp("torch_csr")), mm),
        ("vector@torch_csr", lambda: (vec(), sp("torch_csr")), mm),
        ("dense@stensor", lambda: (_dense(k, n), sp("stensor")), mm),
        ("torch_coo@dense", lambda: (sp("torch_coo"), _dense(n, k)), mm),
        ("stensor_oo@dense", lambda: (sp("stensor_oo"), _dense(n, k)), mm),
        ("torch_csr@torch_csr", lambda: (sp("torch_csr"), sp("torch_csr", 4)), mm),
        ("stensor@stensor", lambda: (sp("stensor"), sp("stensor", 4)), mm),
        ("format=dd", lambda: (sp("stensor"), sp("stensor", 4)), lambda a, b: mm(a, b, format="dd")),
        (
            "einsum",
            lambda: (sp("stensor"), _dense(n, k)),
            lambda a, b: scorch.einsum("ij,jk->ik", a, b, format="dd"),
        ),
        ("recorded x@A", lambda: (_dense(k, n), grad(sp("torch_csr"))), mm),
        ("recorded A@x", lambda: (grad(sp("torch_csr")), _dense(n, k)), mm),
    ]


def _form(x):
    if isinstance(x, STensor):
        return ("STensor", str(x.format), tuple(x.mode_order), tuple(x.shape), x.dtype)
    return ("Tensor", x.layout, tuple(x.shape), x.dtype)


def _values(x):
    if isinstance(x, STensor):
        x = x.to_torch()
    x = x.detach()
    return (x.to_dense() if x.layout != torch.strided else x).numpy()


_IDS = [name for name, _, _ in _calls(4, False)]

_REFUSALS = {
    "tiling_sparse_result_unsupported": "the unforced result is sparse; S writes dense results",
    "sparse_index_affine_tiling_unsupported": "a COO operand: S tiles i, a coordinate level",
    "schedule_without_sparse_operand": "a torch CSC operand is converted dense, or two are torch's",
    "schedule_loop_order_incomplete": "a vector times a matrix has two indices; S orders three",
}
_STENSOR_ROWS = {"stensor@dense", "dense@stensor", "format=dd", "einsum"}
_TORCH_CSR_ROWS = {"torch_csr@dense", "dense@torch_csr", "recorded x@A", "recorded A@x"}


def _must_run(name, transposed_storage):
    return name in _STENSOR_ROWS or (name in _TORCH_CSR_ROWS and not transposed_storage)


@pytest.mark.parametrize("order", [False, True], ids=["stored_order", "transposed_storage"])
@pytest.mark.parametrize("n", [48, 300])
@pytest.mark.parametrize("index", range(len(_IDS)), ids=_IDS)
def test_a_forced_schedule_keeps_the_unforced_result(index, n, order):
    name, operands, product = _calls(n, order)[index]
    try:
        want = product(*operands())
    except RuntimeError as error:
        # A torch built without MKL (the M5's) has no CSR x CSR product, which two torch
        # CSC operands reach; with MKL (sapling's) this row runs.
        if "without MKL" not in str(error):
            raise
        pytest.skip("this torch has no CSR x CSR product")
    forced_operands = operands()
    try:
        with scorch.schedule_force(S):
            got = product(*forced_operands)
    except Exception as error:  # noqa: BLE001 - the refusal is the result checked
        code = defect_code(error)
        sparse_result = (
            not want.format.is_dense()
            if isinstance(want, STensor)
            else want.layout != torch.strided
        )
        detail = (name, _form(want), type(error).__name__, code, str(error)[:300])
        assert code in _REFUSALS and not _must_run(name, order), detail
        if code == "tiling_sparse_result_unsupported":
            assert sparse_result, detail
        elif sparse_result:
            # Two torch CSC operands are torch's product: matmul's first tier hands torch
            # every pair of torch tensors neither of which is COO or CSR, and refuses a
            # schedule there, before any result format is chosen.
            assert code == "schedule_without_sparse_operand", detail
            assert name == "torch_csr@torch_csr" and order, detail
        return
    assert _form(got) == _form(want), name
    w = _values(want)
    assert np.abs(_values(got) - w).max() <= 1e-12 * max(1.0, np.abs(w).max()), name


def test_a_sparse_result_refuses_the_strip_mined_schedule_with_its_code():
    """S's kernel cannot write a sparse result, and the forced call keeps the unforced
    format, so it refuses with the scheduler's code rather than returning another format."""
    a, b = _sparse(48, "stensor", False), _sparse(48, "stensor", False, seed=4)
    with scorch.schedule_force(S), pytest.raises(Exception) as caught:
        ops.matmul(a, b)
    assert defect_code(caught.value) == "tiling_sparse_result_unsupported"


# A forced forward's layout (autograd._scheduled_out): the strides of the Function's own
# product, which follow the dense operand's layout, and never a view.


def _dense_operand(shape, layout, dtype, seed=1):
    """A dense matrix of ``shape``: contiguous, or the transpose of a contiguous one (not a
    view itself, so that it can be a leaf that requires grad)."""
    rows, cols = shape
    if layout == "transposed":
        return rhs(cols, rows, seed).to(dtype).t().detach()
    return rhs(rows, cols, seed).to(dtype)


def _close_at(got, want, dtype):
    got, want = got.detach(), want.detach()
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    assert (got - want).abs().max() <= tol * max(1.0, want.abs().max())


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("width", [K, 1])
@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
@pytest.mark.parametrize("planned", [False, True], ids=["unplanned", "planned"])
@pytest.mark.parametrize("left", [True, False], ids=["A@b", "x@A"])
def test_a_recorded_forced_product_has_the_unforced_strides(
    generated, left, planned, layout, width, dtype
):
    """A recorded product's forced forward (_SparseMM's A @ b, _SparseXA's x @ A) has the
    strides its unforced forward has, on every route that serves the unforced call (the
    Python Functions, or the C entry point's planned ones), and is not a view: column-major
    for a transposed b, and for an x that is not contiguous. A schedule changes the kernel,
    not the result's layout. (A matrix times a vector refuses under the schedule:
    test_matrix_vector_refuses_with_its_code.)"""
    a = torch.from_numpy(dense_a()).to(dtype).to_sparse_csr()
    dense = _dense_operand((N, width) if left else (width, N), layout, dtype)

    def product():
        return ops.matmul(a, dense) if left else ops.matmul(dense, a)

    if planned:
        for _ in range(3):
            product()
    dense.requires_grad_(True)
    want = product()
    del generated[:]
    with scorch.schedule_force(S):
        got = product()
    assert generated == [S]
    assert got.stride() == want.stride(), (got.stride(), want.stride())
    assert got._base is None
    _close_at(got, want, dtype)


def _laid(value, laid):
    """``value`` laid out as a generated kernel's product could be: a new row-major or
    column-major tensor, a contiguous view of a larger buffer, or every other element of
    one (a view whose elements are not adjacent)."""
    if laid == "row_major":
        return value.clone(memory_format=torch.contiguous_format)
    if laid == "column_major":
        return value.t().contiguous().t()
    if laid == "contiguous_view":
        buffer = torch.empty(value.numel() + 1, dtype=value.dtype)
        return buffer[1:].view(value.shape).copy_(value)
    buffer = torch.empty(*value.shape[:-1], 2 * value.shape[-1], dtype=value.dtype)
    return buffer[..., ::2].copy_(value)


_LAIDS = ("row_major", "column_major", "contiguous_view", "strided_view")


def _layout_cases():
    shapes = [(5, 7), (1, 7), (5, 1)]
    for route in ("A@b", "x@A"):
        for rows, cols in shapes:
            for width in (3, 1):
                for layout in ("contiguous", "transposed"):
                    for laid in _LAIDS:
                        yield pytest.param(
                            route, (rows, cols), width, layout, laid,
                            id=f"{route}-{rows}x{cols}-w{width}-{layout}-{laid}",
                        )
    for route in ("A@v", "v@A"):
        for rows, cols in shapes:
            for laid in _LAIDS[:1] + _LAIDS[2:]:  # a vector has no column-major layout
                yield pytest.param(
                    route, (rows, cols), None, "contiguous", laid,
                    id=f"{route}-{rows}x{cols}-{laid}",
                )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("route,a_shape,width,layout,laid", list(_layout_cases()))
def test_a_forced_forward_has_the_unforced_strides_and_copies_only_out_of_order(
    route, a_shape, width, layout, laid, dtype
):
    """Each Function's forced forward (_SparseMM for A @ b and A @ v, _SparseMMT for v @ A,
    _SparseXA for x @ A), handed a product in each layout a generated kernel could return,
    has the strides the unforced forward has and is not a view, and shares the product's
    memory exactly where the product's elements already lie in that order: no copy where
    none is needed, one where one is. Matrices with one row or one column are included,
    where the unforced layout depends on whether the transpose copies (float32 always,
    float64 not). The forward is a stand-in for ops.matmul (``_differentiable``'s
    ``ordinary``), so that the vector routes, which refuse under a real schedule, and every
    layout are reached."""
    rows, cols = a_shape
    rng = np.random.default_rng(3)
    a_values = np.where(rng.random(a_shape) < 0.6, rng.uniform(-1, 1, a_shape), 0.0)
    a = torch.from_numpy(a_values).to(dtype).to_sparse_csr()
    if route == "A@b":
        dense = _dense_operand((cols, width), layout, dtype)
    elif route == "x@A":
        dense = _dense_operand((width, rows), layout, dtype)
    else:
        dense = rhs(1, cols if route == "A@v" else rows).to(dtype)[0].clone()
    left = route in ("A@b", "A@v")
    operands = (a, dense) if left else (dense, a)
    value = a.to_dense() @ dense if left else dense @ a.to_dense()
    want = scorch_autograd._differentiable(*operands)
    made = []

    def ordinary(*_primals):
        made.append(_laid(value, laid))
        return made[-1]

    with scorch.schedule_force(S):
        got = scorch_autograd._differentiable(*operands, ordinary)
    assert len(made) == 1
    product = made[0]
    assert got.stride() == want.stride(), (got.stride(), want.stride())
    assert got._base is None
    assert torch.equal(got, value)
    pairs = zip(product.shape, product.stride(), want.stride())
    in_order = all(s == w for n, s, w in pairs if n > 1)
    shared = got.untyped_storage().data_ptr() == product.untyped_storage().data_ptr()
    assert shared == in_order, ("shared" if shared else "copied", product.stride())
