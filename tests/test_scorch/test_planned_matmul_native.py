"""scorch.matmul's planned fast path in C (csrc/planned_matmul.h).

The native callable must serve exactly the calls the Python probe in ops.matmul served and pass
every other call to the Python function untouched. test_dispatch_plans.py already holds the plan
contract (installation, declines, withdrawal, generations, bitwise agreement) and runs through
scorch.matmul, so it now runs through this. What is specific to the native path, and what this
file checks:

  * WHO served. A plan can serve from the native path or from the Python probe, and the result
    is bit-identical either way -- so an answer check cannot tell whether the native path did
    anything. `sys.setprofile` can: a call the native path serves enters no Python `matmul`
    frame, and a call it hands over enters exactly one.
  * every hand-over case reaches the Python function with its arguments intact, and produces
    what the Python function produces;
  * decline accounting still runs once per declined call, so withdrawal happens on schedule;
  * the callable is still `ops.matmul` for tracing, pickling and introspection.
"""

import inspect
import pickle
import sys
import threading

import pytest
import torch

import scorch
import scorch_ops
from scorch import STensor, ops, plan

pytestmark = pytest.mark.skipif(
    not hasattr(scorch_ops, "install_planned_matmul"),
    reason="scorch_ops predates the native planned matmul",
)


def _operand(rows=120, cols=90, density=0.1, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    dense = torch.rand(rows, cols, generator=g, dtype=dtype)
    dense[torch.rand(rows, cols, generator=g) > density] = 0
    return dense, STensor.from_torch(dense.to_sparse_csr())


class _MatmulFrames:
    """Counts entries into the Python matmul's frame while active."""

    def __init__(self):
        self.code = ops._matmul_python.__code__
        self.calls = 0

    def _hook(self, frame, event, arg):
        if event == "call" and frame.f_code is self.code:
            self.calls += 1

    def __enter__(self):
        self._prev = sys.getprofile()
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *exc):
        sys.setprofile(self._prev)


def _planned(st, b, calls=3):
    for _ in range(calls):
        scorch.matmul(st, b)
    plans = plan.plans_of(st)
    assert (
        len(plans) == 1
    ), "the operand should hold exactly one plan after repeated calls"
    return next(iter(plans.values()))


def test_matmul_is_the_native_callable_and_still_ops_matmul():
    assert scorch.matmul is ops.matmul
    assert type(ops.matmul).__name__ == "builtin_function_or_method"
    assert ops.matmul.__module__ == "scorch.ops" and ops.matmul.__name__ == "matmul"
    assert pickle.loads(pickle.dumps(ops.matmul)) is ops.matmul
    assert list(inspect.signature(ops.matmul).parameters) == ["a", "b", "kwargs"]
    assert (
        ops._matmul_python.__doc__ and ops._matmul_python.__doc__ in ops.matmul.__doc__
    )


def test_a_planned_call_enters_no_python_frame_and_is_served_by_the_plan():
    dense, st = _operand()
    b = torch.rand(90, 8)
    p = _planned(st, b)
    before = p.served
    with _MatmulFrames() as frames:
        out = scorch.matmul(st, b)
    assert frames.calls == 0, "the native path should have served this call"
    assert p.served == before + 1
    torch.testing.assert_close(out, dense @ b, rtol=1e-5, atol=1e-5)


def test_the_profile_hook_sees_the_python_frame_when_it_is_entered():
    # The control for the test above: without it, a hook that never fires would pass it.
    dense, st = _operand()
    b = torch.rand(90, 8)
    with _MatmulFrames() as frames:
        ops._matmul_python(st, b)
    assert frames.calls == 1


@pytest.mark.parametrize(
    "case",
    [
        "keyword",
        "parameter_rhs",
        "stensor_rhs",
        "dense_lhs",
        "plans_disabled",
        "new_generation",
    ],
)
def test_every_other_call_reaches_the_python_function_once_and_agrees(case):
    dense, st = _operand(seed=3)
    b = torch.rand(90, 6)
    _planned(st, b)
    args, kwargs, expect = (st, b), {}, dense @ b
    restore = None
    if case == "keyword":
        kwargs = {"format": "dd"}
    elif case == "parameter_rhs":
        args = (st, torch.nn.Parameter(b.clone(), requires_grad=False))
    elif case == "stensor_rhs":
        args = (st, STensor.from_torch(b))
    elif case == "dense_lhs":
        args, expect = (dense, b), dense @ b
    elif case == "plans_disabled":
        restore = plan.enabled()
        plan.set_enabled(False)
    elif case == "new_generation":
        plan.invalidate_all()
    try:
        with _MatmulFrames() as frames:
            out = scorch.matmul(*args, **kwargs)
    finally:
        if restore is not None:
            plan.set_enabled(restore)
    assert frames.calls == 1, case
    out = out.to_torch() if isinstance(out, STensor) else out
    torch.testing.assert_close(out, expect, rtol=1e-5, atol=1e-5)


def test_a_declined_call_is_counted_once_and_withdrawal_comes_on_schedule():
    # Both the native path and the Python probe try the plan on a declined call, so the risk is
    # counting it twice and withdrawing early. A right-hand operand that is the transpose of a
    # contiguous one has the same shape and dtype -- the same key -- and every plan declines it.
    dense, st = _operand(seed=5)
    bt = torch.rand(6, 90).t()
    assert not bt.is_contiguous()
    scorch.matmul(st, bt)  # first sighting
    scorch.matmul(st, bt)  # installs a plan that will only ever decline
    plans = st.__dict__[plan.PLANS_ATTR]
    (key,) = list(plans)
    assert plans[key] is not None and plans[key].served == 0
    for i in range(1, plan.MAX_FRUITLESS_DECLINES + 1):
        with _MatmulFrames() as frames:
            out = scorch.matmul(st, bt)
        assert frames.calls == 1
        torch.testing.assert_close(out, dense @ bt, rtol=1e-5, atol=1e-5)
        assert (
            st.__dict__[plan.DECLINES_ATTR][key] == i
        ), "a declined call must count once"
        withdrawn = plans[key] is None
        assert withdrawn == (i >= plan.MAX_FRUITLESS_DECLINES), i
    with (
        _MatmulFrames() as frames
    ):  # withdrawn: a plain miss, answered by the Python function
        torch.testing.assert_close(
            scorch.matmul(st, bt), dense @ bt, rtol=1e-5, atol=1e-5
        )
    assert frames.calls == 1


def test_errors_are_the_python_functions_errors():
    _, st = _operand()
    _planned(st, torch.rand(90, 8))
    wrong = torch.rand(
        91, 8
    )  # contraction mismatch: the plan declines, the ordinary path raises
    with pytest.raises(Exception) as native_err:
        scorch.matmul(st, wrong)
    with pytest.raises(Exception) as python_err:
        ops._matmul_python(st, wrong)
    assert type(native_err.value) is type(python_err.value)
    assert str(native_err.value) == str(python_err.value)


def test_launching_threads_each_get_the_right_answer():
    dense, st = _operand(rows=400, cols=300, density=0.05, seed=9)
    bs = [
        torch.rand(300, 4, generator=torch.Generator().manual_seed(i)) for i in range(4)
    ]
    for b in bs:
        _planned(st, b, calls=2)
    errors = []

    def run(i):
        for _ in range(50):
            out = scorch.matmul(st, bs[i])
            if not torch.allclose(out, dense @ bs[i], rtol=1e-5, atol=1e-5):
                errors.append(i)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors


def test_tracing_records_ops_matmul_and_the_traced_graph_runs():
    dense, st = _operand()
    b = torch.rand(90, 8)

    def f(a, x):
        return torch.relu(scorch.matmul(a, x)) + 1

    g = torch.fx.symbolic_trace(f)
    targets = [n.target for n in g.graph.nodes if n.op == "call_function"]
    assert targets[0] is ops.matmul
    torch.testing.assert_close(
        g(st, b), torch.relu(dense @ b) + 1, rtol=1e-5, atol=1e-5
    )
