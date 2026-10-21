"""The checks every ``matmul`` call makes before it looks for a kernel.

Two run on every call: the operands must hold one value dtype, and an explicit or
context-forced schedule needs a sparse operand.  Both used to go through helper frames
on every call, which cost 0.7 us of a 41 us warm ``STensor`` product on mkt1 (beat-mkl,
job sc8r).  For the pairs a warm call makes (an ``STensor`` or a torch CSR tensor times
a ``torch.Tensor``) and for a call with no keyword arguments, ``matmul`` now makes them
inline.  These tests pin what the checks decide on both paths, and that the inline
path does not enter the helpers.

A product that ``einsum`` runs (no prebuilt kernel takes it) used to have ``matmul``
make an options snapshot on every call and hand it on: 60 us of a 250 us warm COO
times dense product on sapling (job 82748).  ``einsum`` now makes it only where it
needs one, so a warm product makes none, and a forced schedule still reaches it.
"""

import pytest
import torch

import scorch
from scorch import ops
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.scheduler import Schedule, schedule_force
from scorch.exceptions import OperandDtypeMismatchError
from tests.test_scorch.level_array_support import stored


def _matrix(dtype, seed=3):
    generator = torch.Generator().manual_seed(seed)
    keep = torch.rand((6, 5), generator=generator) < 0.5
    return (torch.rand((6, 5), generator=generator, dtype=torch.float64) * keep).to(
        dtype
    )


def _dense(shape, dtype, seed=4):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=generator, dtype=torch.float64).to(dtype)


def _fail(*args, **kwargs):
    raise AssertionError("the inline path entered a helper")


LEFT = {
    "stensor_ds": lambda dtype: stored(_matrix(dtype), "ds", "A"),
    "stensor_oo": lambda dtype: stored(_matrix(dtype), "oo", "A"),
    "torch_csr": lambda dtype: _matrix(dtype).to_sparse_csr(),
}
RIGHT_SHAPES = [(5,), (5, 3)]


@pytest.mark.parametrize("left", sorted(LEFT))
@pytest.mark.parametrize("right_shape", RIGHT_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_one_dtype_runs_without_the_helpers(monkeypatch, left, right_shape, dtype):
    a = LEFT[left](dtype)
    b = _dense(right_shape, dtype)
    monkeypatch.setattr(ops, "_require_one_product_dtype", _fail)
    monkeypatch.setattr(ops, "_matmul_compile_policy", _fail)

    result = scorch.matmul(a, b)

    expected = torch.matmul(_matrix(torch.float64), b.to(torch.float64))
    got = result.to_dense() if result.layout != torch.strided else result
    assert got.dtype == dtype
    torch.testing.assert_close(got.to(torch.float64), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("left", sorted(LEFT))
@pytest.mark.parametrize("right_shape", RIGHT_SHAPES)
@pytest.mark.parametrize(
    ("left_dtype", "right_dtype"),
    [(torch.float32, torch.float64), (torch.float64, torch.float32)],
)
def test_two_dtypes_raise_through_the_helper(
    left, right_shape, left_dtype, right_dtype
):
    a = LEFT[left](left_dtype)
    b = _dense(right_shape, right_dtype)
    with pytest.raises(OperandDtypeMismatchError) as raised:
        scorch.matmul(a, b)
    assert raised.value.code == "operand_dtype_mismatch"
    assert "matmul operands must have the same value dtype" in str(raised.value)


@pytest.mark.parametrize("same", [True, False])
def test_a_tensor_subclass_operand_takes_the_helper(monkeypatch, same):
    a = stored(_matrix(torch.float64), "ds", "A")
    b = torch.nn.Parameter(
        _dense((5, 3), torch.float64 if same else torch.float32), requires_grad=False
    )
    calls = []
    helper = ops._require_one_product_dtype
    monkeypatch.setattr(
        ops,
        "_require_one_product_dtype",
        lambda *args: calls.append(args) or helper(*args),
    )
    if same:
        scorch.matmul(a, b)
    else:
        with pytest.raises(OperandDtypeMismatchError):
            scorch.matmul(a, b)
    assert len(calls) == 1


def _dense_pairs():
    a, b = _dense((4, 5), torch.float64, 5), _dense((5, 3), torch.float64, 6)
    return [
        ("torch", a, b, torch.matmul(a, b)),
        ("stensor", stored(a, "dd", "A"), stored(b, "dd", "B"), torch.matmul(a, b)),
    ]


@pytest.mark.parametrize("pair", [0, 1], ids=["torch", "stensor"])
def test_a_forced_schedule_refuses_a_dense_product(pair):
    _, a, b, _ = _dense_pairs()[pair]
    with schedule_force(Schedule(tag="forced")):
        with pytest.raises(NotImplementedError, match="require a sparse operand"):
            scorch.matmul(a, b)


@pytest.mark.parametrize("pair", [0, 1], ids=["torch", "stensor"])
def test_a_schedule_argument_refuses_a_dense_product(pair):
    _, a, b, _ = _dense_pairs()[pair]
    with pytest.raises(NotImplementedError, match="require a sparse operand"):
        scorch.matmul(a, b, schedule=Schedule(tag="argument"))


@pytest.mark.parametrize("pair", [0, 1], ids=["torch", "stensor"])
def test_no_schedule_runs_a_dense_product_without_the_policy(monkeypatch, pair):
    _, a, b, expected = _dense_pairs()[pair]
    monkeypatch.setattr(ops, "_matmul_compile_policy", _fail)
    torch.testing.assert_close(scorch.matmul(a, b), expected)


def test_keyword_arguments_still_resolve_the_policy(monkeypatch):
    a = stored(_matrix(torch.float64), "ds", "A")
    b = _dense((5, 3), torch.float64)
    calls = []
    policy = ops._matmul_compile_policy
    monkeypatch.setattr(
        ops,
        "_matmul_compile_policy",
        lambda kwargs: calls.append(dict(kwargs)) or policy(kwargs),
    )
    scorch.matmul(a, b, format="dd")
    assert calls == [{"format": "dd"}]


def _generated_product():
    """COO times dense in float64, which no prebuilt kernel takes: einsum runs it."""

    a = stored(_matrix(torch.float64), "oo", "A")
    b = _dense((5, 3), torch.float64)
    return a, b, torch.matmul(_matrix(torch.float64), b)


def test_a_warm_generated_product_makes_no_options_snapshot(monkeypatch):
    a, b, expected = _generated_product()
    torch.testing.assert_close(scorch.matmul(a, b), expected)

    routed = []
    einsum = ops.einsum
    monkeypatch.setattr(
        ops, "einsum", lambda *args, **kw: routed.append(kw) or einsum(*args, **kw)
    )
    made = []
    parse = CompileOptions.from_environment.__func__
    monkeypatch.setattr(
        CompileOptions,
        "from_environment",
        classmethod(
            lambda cls, *args, **kw: made.append(kw) or parse(cls, *args, **kw)
        ),
    )
    for _ in range(3):
        torch.testing.assert_close(scorch.matmul(a, b), expected)
    assert len(routed) == 3
    assert all("_compile_options" not in kw for kw in routed)
    assert made == []


def test_a_forced_schedule_reaches_the_generated_product(monkeypatch):
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
    a, b, expected = _generated_product()
    schedule = Schedule(loop_order=("i", "j", "k"), tag="forced-generated-product")
    with schedule_force(schedule):
        torch.testing.assert_close(scorch.matmul(a, b), expected)
    assert any(schedule.cache_key in key for key in ops._einsum_dispatch_cache)
