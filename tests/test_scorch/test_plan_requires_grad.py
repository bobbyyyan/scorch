"""A plan never answers a grad-mode product on an STensor that wants a gradient.

The C entry point (``csrc/planned_matmul.h``) serves a plan without asking
``autograd.py``. So on each call it would serve, it reads the STensor's current
``requires_grad`` and tests it against grad mode, as ``planned_matmul_wants_grad``
does for a torch operand: in grad mode a flagged operand goes to the Python
function, which records the product; under no_grad its plan serves it, so a
trained sparse weight evaluated for inference keeps its planned path. The flag's
setter drops nothing.

``autograd.wanted`` is wrapped to count the calls that reach it: a call the C entry
point serves from a plan never does, so a count of zero on a grad-mode call means a
plan answered it. ``test_unflagged_grad_mode_call_is_served_in_c`` is the positive
control that shows the count sees a served call.
"""

import pytest
import torch

import scorch
import scorch.autograd as scorch_autograd
from scorch import STensor
from scorch import plan as plan_mod
from scorch.plan import plans_of, refused_of


def random_csr(rows, cols, density, dtype=torch.float32, seed=0):
    """A CSR STensor and the dense tensor it came from."""
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((rows, cols), generator=generator) < density
    dense = (torch.rand((rows, cols), generator=generator) * mask).to(dtype)
    return STensor.from_torch(dense.to_sparse_csr()), dense


def dense_rhs(cols, n, dtype=torch.float32, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((cols, n), generator=generator).to(dtype)


def as_tensor(result):
    out = result if isinstance(result, torch.Tensor) else result.to_torch(False)
    return out.detach()


def plain_product(dense, b):
    """The same product on a new, unflagged operand with plans off: the bitwise
    reference for a call computed on the ordinary path."""
    was_enabled = plan_mod.enabled()
    plan_mod.set_enabled(False)
    try:
        return as_tensor(scorch.matmul(STensor.from_torch(dense.to_sparse_csr()), b))
    finally:
        plan_mod.set_enabled(was_enabled)


def float64_product(dense, b):
    return (dense.double() @ b.double()).float()


@pytest.fixture(autouse=True)
def plans_on():
    """Every test starts with installation enabled, whatever the env says."""
    was_enabled = plan_mod.enabled()
    plan_mod.set_enabled(True)
    yield
    plan_mod.set_enabled(was_enabled)


@pytest.fixture
def wanted_calls(monkeypatch):
    """Count the calls that reach autograd.wanted; the original still decides."""
    original = scorch_autograd.wanted
    calls = [0]

    def counting(a, b):
        calls[0] += 1
        return original(a, b)

    monkeypatch.setattr(scorch_autograd, "wanted", counting)
    return calls


def planned(a, b, calls=3):
    """Make the product of `a` and `b` planned, under no_grad; return its plan."""
    with torch.no_grad():
        for _ in range(calls):
            scorch.matmul(a, b)
    (plan,) = plans_of(a).values()
    return plan


def test_unflagged_grad_mode_call_is_served_in_c(wanted_calls):
    """Positive control: a planned grad-mode call on an operand that wants no
    gradient is served in C, and the count does not move."""
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    plan = planned(a, b)
    before, served = wanted_calls[0], plan.served
    out = as_tensor(scorch.matmul(a, b))
    assert wanted_calls[0] == before
    assert plan.served == served + 1
    torch.testing.assert_close(out, plain_product(dense, b), atol=0, rtol=0)


def test_flagged_grad_mode_call_reaches_autograd(wanted_calls):
    """The case a plan used to answer: planned under no_grad, flagged, then a
    grad-mode call."""
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    plan = planned(a, b)
    a.requires_grad = True
    assert list(plans_of(a).values()) == [plan], "setting the flag dropped the plan"
    before = wanted_calls[0]
    out = scorch.matmul(a, b)
    assert wanted_calls[0] > before, "a plan served a grad-mode call without autograd"
    torch.testing.assert_close(
        as_tensor(out), float64_product(dense, b), atol=1e-5, rtol=1e-5
    )


def test_flagged_operand_is_served_by_its_plan_under_no_grad(wanted_calls):
    a, dense = random_csr(64, 64, 0.05)
    a.requires_grad = True
    b = dense_rhs(64, 8)
    plan = planned(a, b)
    assert refused_of(a) == set()
    before, served = wanted_calls[0], plan.served
    with torch.no_grad():
        out = as_tensor(scorch.matmul(a, b))
    assert plan.served == served + 1, "the flagged operand's plan did not serve"
    assert wanted_calls[0] == before, "a no_grad planned call left the C entry"
    torch.testing.assert_close(out, plain_product(dense, b), atol=0, rtol=0)


def test_inference_mode_counts_as_no_grad():
    a, dense = random_csr(64, 64, 0.05)
    a.requires_grad = True
    b = dense_rhs(64, 8)
    plan = planned(a, b)
    served = plan.served
    with torch.inference_mode():
        out = as_tensor(scorch.matmul(a, b))
    assert plan.served == served + 1
    torch.testing.assert_close(out, plain_product(dense, b), atol=0, rtol=0)


def test_clearing_the_flag_serves_grad_mode_calls_in_c_again(wanted_calls):
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    plan = planned(a, b)
    a.requires_grad = True
    scorch.matmul(a, b)
    a.requires_grad = False
    before, served = wanted_calls[0], plan.served
    out = as_tensor(scorch.matmul(a, b))
    assert wanted_calls[0] == before
    assert plan.served == served + 1
    torch.testing.assert_close(out, plain_product(dense, b), atol=0, rtol=0)


def test_setting_requires_grad_keeps_the_gradient_structure():
    """The setter drops nothing: plans and autograd.py's structure both stay."""
    a, _ = random_csr(32, 32, 0.1)
    marker = object()
    a.__dict__[plan_mod.GRAD_ATTR] = marker
    a.requires_grad = True
    assert a.__dict__.get(plan_mod.GRAD_ATTR) is marker
    a.requires_grad = False
    assert a.__dict__.get(plan_mod.GRAD_ATTR) is marker


def test_unflagged_operand_is_still_planned():
    """The control: an operand that wants no gradient is planned on its second call
    and served on its third, bitwise equal to the ordinary path."""
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    for _ in range(2):
        scorch.matmul(a, b)
    (plan,) = plans_of(a).values()
    before = plan.served
    out = as_tensor(scorch.matmul(a, b))
    assert plan.served == before + 1
    torch.testing.assert_close(out, plain_product(dense, b), atol=0, rtol=0)
