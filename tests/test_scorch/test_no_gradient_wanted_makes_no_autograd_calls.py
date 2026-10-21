"""A call through which no gradient is wanted makes no call into autograd.py.

fast_transpose, sparse_linear_fm and einsum each record a derivative when a gradient is
wanted through them, and each decides whether one is from grad mode and the operands'
flags, read in place, then from forward-mode AD: so a call under ``torch.no_grad``, or
in grad mode with no operand asking for a gradient, reaches none of autograd.py's
functions. Counted with sys.setprofile, as test_dense_product_made_shaped.py counts a
training step's calls.
"""

import collections
import gc
import os
import sys

import pytest
import torch

import scorch
import scorch_ops
from scorch import STensor
from scorch import ops
from scorch.exceptions import ScorchError

_AUTOGRAD_PY = os.path.join("scorch", "autograd.py")


def _autograd_calls(call):
    """The Python calls ``call()`` makes in autograd.py, by qualified name."""
    counts = collections.Counter()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_filename.endswith(_AUTOGRAD_PY):
            counts[frame.f_code.co_qualname] += 1

    gc.collect()
    gc.disable()
    sys.setprofile(profile)
    try:
        call()
    finally:
        sys.setprofile(None)
        gc.enable()
    return dict(counts)


def _sparse(rows, cols, seed):
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((rows, cols), generator=generator) < 0.3
    return torch.rand((rows, cols), generator=generator) * mask


def _calls(x_asks):
    """Each operation's call; its dense operand asks for a gradient when ``x_asks``."""
    x = torch.rand(9, 7).requires_grad_(x_asks)
    weight = scorch.from_csr(_sparse(16, 12, seed=1).to_sparse_csr(), "w")
    x_fm = torch.rand(12, 5).requires_grad_(x_asks)
    bias = torch.rand(16)
    a = scorch.from_csr(_sparse(6, 9, seed=2).to_sparse_csr(), "a")
    return {
        "fast_transpose": lambda: scorch.fast_transpose(x),
        "sparse_linear_fm": lambda: scorch.sparse_linear_fm(x_fm, weight, bias),
        "einsum": lambda: scorch.einsum("ij,jk->ik", a, x, format="dd"),
    }


@pytest.mark.parametrize("mode", ["no_grad", "grad mode, no operand asking"])
@pytest.mark.parametrize("name", ["fast_transpose", "sparse_linear_fm", "einsum"])
def test_a_call_through_which_no_gradient_is_wanted_makes_no_autograd_call(name, mode):
    # fast_transpose tests for a gradient only on its native kernel's path
    assert getattr(scorch_ops, "scorch_transpose_2d_float", None) is not None
    call = _calls(x_asks=False)[name]
    with torch.set_grad_enabled(mode != "no_grad"):
        for _ in range(2):  # the first call builds what the operation keeps
            call()
        made = _autograd_calls(call)
    assert made == {}, f"{name}, {mode}: autograd.py calls {made}"


@pytest.mark.parametrize("name", ["fast_transpose", "sparse_linear_fm", "einsum"])
def test_a_call_through_which_a_gradient_is_wanted_is_recorded(name):
    """The positive control: with the dense operand asking for a gradient, each call
    records its derivative (a grad_fn on the result), so the test above counts calls on
    the path that decides it; and the count sees autograd.py, where sparse_linear_fm's
    derivative is recorded (autograd.linear_fm)."""
    call = _calls(x_asks=True)[name]
    call()
    result = []
    made = _autograd_calls(lambda: result.append(call()))
    if name == "sparse_linear_fm":
        assert made.get("linear_fm") == 1, made
    assert result[0].requires_grad and result[0].grad_fn is not None, name


# einsum reads its operands' flags in place, in each call form torch.einsum takes (the
# selection _einsum_operands makes): the operands are the positional arguments, the one
# list of them, or every other argument of the sublist form before its output sublist.
_EINSUM_FORMS = {
    "explicit": lambda p, q: scorch.einsum("ij,jk->ik", p, q, format="dd"),
    "implicit": lambda p, q: scorch.einsum("ij,jk", p, q, format="dd"),
    "list": lambda p, q: scorch.einsum("ij,jk->ik", [p, q], format="dd"),
    "sublist": lambda p, q: scorch.einsum(p, [0, 1], q, [1, 2], format="dd"),
    "sublist with output": lambda p, q: scorch.einsum(
        p, [0, 1], q, [1, 2], [0, 2], format="dd"
    ),
}


@pytest.mark.parametrize("position", ["sparse first", "dense first"])
@pytest.mark.parametrize("form", list(_EINSUM_FORMS))
def test_an_einsum_operand_asking_a_gradient_is_seen_in_every_call_form(
    form, position, monkeypatch
):
    """A sparse operand and a dense one that requires grad, in either position and in
    each call form: the call is recorded (ops._einsum_recorded, which einsum_grad's
    derivative or its coded refusal records). A form the compiler computes gets a
    grad_fn and the float64 reference's gradient; a form it refuses is refused with the
    code it has under torch.no_grad, so the decision does not change what a call does.
    """
    a_dense = _sparse(6, 9, seed=3).double()
    a = scorch.from_csr(a_dense.to_sparse_csr(), "a")
    x = torch.rand(
        (9, 4) if position == "sparse first" else (4, 6), dtype=torch.float64
    )
    p, q = (a, x) if position == "sparse first" else (x, a)
    p_ref, q_ref = (a_dense, x) if position == "sparse first" else (x, a_dense)
    call = _EINSUM_FORMS[form]

    with torch.no_grad():
        try:
            call(p, q)
            refused = None
        except ScorchError as error:
            refused = error.code
    # the explicit form is the one the compiler computes in every position
    assert form != "explicit" or refused is None, refused

    recorded = []
    original = ops._einsum_recorded

    def spy(*args, **kwargs):
        recorded.append(args[0] if isinstance(args[0], str) else "<sublist form>")
        return original(*args, **kwargs)

    monkeypatch.setattr(ops, "_einsum_recorded", spy)
    x.requires_grad_()
    if refused is not None:
        with pytest.raises(ScorchError) as error:
            call(p, q)
        assert error.value.code == refused, (form, position)
    else:
        out = call(p, q)
        out = out.to_torch(in_place=False) if type(out) is STensor else out
        assert out.grad_fn is not None, (form, position)
        g = torch.rand(out.shape, dtype=torch.float64)
        (out * g).sum().backward()
        x_ref = x.detach().clone().requires_grad_()
        p_r, q_r = (p_ref, x_ref) if position == "sparse first" else (x_ref, q_ref)
        want = torch.einsum("ij,jk->ik", p_r, q_r)
        (want * g).sum().backward()
        torch.testing.assert_close(out.detach(), want.detach())
        torch.testing.assert_close(x.grad, x_ref.grad)
    assert len(recorded) == 1, (form, position, recorded)
