"""A cached call plan must be indistinguishable from the path it replaces.

``scorch.plan`` lets a repeated CSR x dense product skip resolution, the tiling
selector, index validation and argument marshalling by holding them in a native
plan object (``csrc/plan.h``). That is only legitimate if a planned call returns
exactly what the ordinary dispatch would have returned, if the plan declines
every call it was not built for, and if it cannot outlive the structure it
describes. These tests pin all three, plus the rule that a plan is never
installed for an operand that has only been seen once.

The comparisons here are bitwise, not approximate: a plan runs the same kernel on
the same buffers, so anything other than bit-identical output is a defect.
"""

import copy
import pickle

import numpy as np
import pytest
import torch

import scorch
import scorch_ops
from scorch import STensor
from scorch import plan as plan_mod
from scorch.plan import (
    MAX_FRUITLESS_DECLINES,
    MAX_PLANS_PER_OPERAND,
    plans_of,
    refused_of,
)


def csr_stensor(dense):
    return STensor.from_torch(dense.to_sparse_csr())


def random_csr(rows, cols, density, dtype=torch.float32, seed=0):
    """A CSR STensor and the dense tensor it came from."""
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((rows, cols), generator=generator) < density
    dense = (torch.rand((rows, cols), generator=generator) * mask).to(dtype)
    return csr_stensor(dense), dense


def dense_rhs(cols, n, dtype=torch.float32, seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((cols, n), generator=generator).to(dtype)


def matmul_without_plans(a, b):
    """The same product with plans switched off, for a bitwise reference."""
    plan_mod.forget(a)
    was_enabled = plan_mod.enabled()
    plan_mod.set_enabled(False)
    try:
        out = scorch.matmul(a, b)
    finally:
        plan_mod.set_enabled(was_enabled)
        plan_mod.forget(a)
    return out


def as_tensor(result):
    return result if isinstance(result, torch.Tensor) else result.to_torch(False)


@pytest.fixture(autouse=True)
def plans_on():
    """Every test starts with installation enabled, whatever the env says."""
    was_enabled = plan_mod.enabled()
    plan_mod.set_enabled(True)
    yield
    plan_mod.set_enabled(was_enabled)


@pytest.fixture(scope="module")
def generated_kernels():
    """Skip on a host whose toolchain cannot compile a generated kernel.

    Some of what a plan must stay clear of -- ``use_cache=False``, ``insert``, a
    relayout -- routes through the JIT compiler, and one of the two hosts this
    work is measured on cannot build a generated kernel at all (a libc++ /
    ``-march=native`` problem in the macOS SDK, unrelated to anything here). The
    probe is a real compile, so the skip is a fact about the host rather than a
    way of not testing this.
    """
    probe = csr_stensor(torch.eye(2))
    try:
        scorch.matmul(probe, torch.eye(2), use_cache=False)
    except Exception as exc:  # pragma: no cover - host dependent
        pytest.skip(
            f"this host cannot compile a generated kernel: {type(exc).__name__}"
        )


# --------------------------------------------------------------------------- #
# Installation policy
# --------------------------------------------------------------------------- #


def test_no_plan_after_one_call_and_a_plan_after_two():
    """A single-use operand must not pay for a plan it will never reuse."""
    a, _ = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    scorch.matmul(a, b)
    assert plans_of(a) == {}, "a plan was installed on first sight of the operand"
    scorch.matmul(a, b)
    assert len(plans_of(a)) == 1


def test_plan_serves_every_call_after_installation():
    a, _ = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    for _ in range(5):
        scorch.matmul(a, b)
    (plan,) = plans_of(a).values()
    assert plan.kind == "v2"
    assert plan.served == 3, f"plan served {plan.served} of the last 3 calls"


def test_alternating_free_dimensions_each_get_their_own_plan():
    a, _ = random_csr(128, 128, 0.05)
    wide, narrow = dense_rhs(128, 32), dense_rhs(128, 4)
    for _ in range(3):
        scorch.matmul(a, wide)
        scorch.matmul(a, narrow)
    free_dims = sorted(p.free_dim for p in plans_of(a).values())
    assert free_dims == [4, 32]


def test_plan_count_is_bounded():
    """Sweeping the free dimension cannot accumulate plans without limit."""
    a, _ = random_csr(64, 64, 0.05)
    for n in range(1, MAX_PLANS_PER_OPERAND + 6):
        b = dense_rhs(64, n)
        scorch.matmul(a, b)
        scorch.matmul(a, b)
    assert len(plans_of(a)) == MAX_PLANS_PER_OPERAND
    # ...and the products past the bound are still correct, via the ordinary path.
    b = dense_rhs(64, MAX_PLANS_PER_OPERAND + 5)
    torch.testing.assert_close(
        as_tensor(scorch.matmul(a, b)),
        as_tensor(matmul_without_plans(a, b)),
        atol=0,
        rtol=0,
    )


def test_disabled_installs_nothing():
    a, _ = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    plan_mod.set_enabled(False)
    for _ in range(4):
        scorch.matmul(a, b)
    assert plans_of(a) == {}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": "dd"},
        {"output_format": "dd"},
        {"time_dict": {}},
    ],
)
def test_keyword_arguments_never_take_the_plan(kwargs):
    """Keywords change what a call means or must report, so they bypass plans."""
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    for _ in range(3):
        scorch.matmul(a, b)
    (plan,) = plans_of(a).values()
    before = plan.served
    passed = dict(kwargs)
    out = scorch.matmul(a, b, **passed)
    assert plan.served == before, "a keyword call was served by the plan"
    reference = torch.matmul(dense.double(), b.double()).float()
    torch.testing.assert_close(as_tensor(out), reference, atol=1e-3, rtol=1e-3)
    if "time_dict" in kwargs:
        assert "eval_time" in passed["time_dict"], "timing contract broken"


def test_use_cache_false_is_not_served_by_a_plan(generated_kernels):
    """``use_cache=False`` asks for the generic compiler path by name; a plan
    answering it would silently return the prebuilt kernel's result instead."""
    a, dense = random_csr(32, 32, 0.1)
    b = dense_rhs(32, 4)
    for _ in range(3):
        scorch.matmul(a, b)
    (plan,) = plans_of(a).values()
    before = plan.served
    out = scorch.matmul(a, b, use_cache=False)
    assert plan.served == before, "use_cache=False was served by the plan"
    torch.testing.assert_close(
        as_tensor(out),
        torch.matmul(dense.double(), b.double()).float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_products_that_never_get_a_plan():
    """Only CSR x dense-tensor products are planned; the rest are untouched."""
    a, dense = random_csr(32, 32, 0.1)
    sparse_rhs, _ = random_csr(32, 32, 0.1)
    vector = torch.rand(32)
    dense_lhs = STensor.from_torch(torch.rand(8, 32))
    for _ in range(3):
        scorch.matmul(a, sparse_rhs, format="ds")  # SpGEMM
        scorch.matmul(a, vector)  # SpMV
        scorch.matmul(dense_lhs, torch.rand(32, 4))
    assert plans_of(a) == {}
    assert plans_of(sparse_rhs) == {}
    assert plans_of(dense_lhs) == {}


# --------------------------------------------------------------------------- #
# Equivalence with the ordinary path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "rows,cols,density,n,dtype",
    [
        (1, 1, 1.0, 1, torch.float32),
        (7, 5, 0.4, 3, torch.float32),
        (64, 64, 0.05, 1, torch.float32),
        (64, 64, 0.05, 8, torch.float32),
        (64, 96, 0.02, 32, torch.float32),
        (128, 64, 0.5, 16, torch.float32),
        (200, 300, 0.01, 64, torch.float32),
        (64, 64, 0.0, 8, torch.float32),  # every row empty
        (64, 64, 1.0, 4, torch.float32),  # fully dense structure
        (64, 64, 0.05, 8, torch.float64),
        (128, 128, 0.03, 16, torch.float64),
    ],
)
def test_planned_output_is_bit_identical(rows, cols, density, n, dtype):
    a, dense = random_csr(rows, cols, density, dtype=dtype)
    b = dense_rhs(cols, n, dtype=dtype)
    reference = as_tensor(matmul_without_plans(a, b))

    scorch.matmul(a, b)  # first sighting
    scorch.matmul(a, b)  # installs
    planned = as_tensor(scorch.matmul(a, b))
    plans = plans_of(a)
    assert len(plans) == 1, "expected exactly one plan for this product"
    (plan,) = plans.values()
    assert plan.served >= 1

    assert planned.dtype == reference.dtype
    assert planned.shape == reference.shape
    assert planned.requires_grad == reference.requires_grad
    torch.testing.assert_close(planned, reference, atol=0, rtol=0)
    # ...and against an independent float64 computation, so a bug shared by both
    # paths would still be caught.
    expected = torch.matmul(dense.double(), b.double())
    torch.testing.assert_close(
        planned.double(),
        expected,
        atol=1e-6 if dtype == torch.float64 else 1e-3,
        rtol=1e-6 if dtype == torch.float64 else 1e-3,
    )


def test_float64_uses_the_reference_kernel():
    a, _ = random_csr(64, 64, 0.05, dtype=torch.float64)
    b = dense_rhs(64, 8, dtype=torch.float64)
    for _ in range(3):
        scorch.matmul(a, b)
    (plan,) = plans_of(a).values()
    assert plan.kind == "reference"


def test_each_result_is_a_fresh_buffer():
    a, _ = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    outs = [scorch.matmul(a, b) for _ in range(4)]
    pointers = {out.data_ptr() for out in outs}
    assert len(pointers) == len(outs), "a planned call reused an output buffer"
    for out in outs[1:]:
        torch.testing.assert_close(out, outs[0], atol=0, rtol=0)


def test_values_written_between_calls_are_picked_up():
    """The plan pins the structure, never the values."""
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    for _ in range(3):
        scorch.matmul(a, b)
    a.storage.value.mul_(3.0)
    torch.testing.assert_close(
        as_tensor(scorch.matmul(a, b)),
        torch.matmul((dense * 3.0).double(), b.double()).float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_planned_call_inside_inference_mode():
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    for _ in range(3):
        scorch.matmul(a, b)
    with torch.inference_mode():
        out = scorch.matmul(a, b)
    torch.testing.assert_close(
        out.clone(),
        torch.matmul(dense.double(), b.double()).float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_plan_built_inside_inference_mode_serves_outside():
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    with torch.inference_mode():
        scorch.matmul(a, b)
        scorch.matmul(a, b)
    out = as_tensor(scorch.matmul(a, b))
    torch.testing.assert_close(
        out,
        torch.matmul(dense.double(), b.double()).float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_right_hand_operand_requiring_grad_matches_the_ordinary_path():
    a, _ = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8).requires_grad_(True)
    reference = as_tensor(matmul_without_plans(a, b))
    scorch.matmul(a, b)
    scorch.matmul(a, b)
    planned = as_tensor(scorch.matmul(a, b))
    assert planned.requires_grad == reference.requires_grad
    torch.testing.assert_close(planned, reference, atol=0, rtol=0)


# --------------------------------------------------------------------------- #
# Declining
# --------------------------------------------------------------------------- #


def _install(rows=64, cols=64, n=8, density=0.05, dtype=torch.float32):
    a, dense = random_csr(rows, cols, density, dtype=dtype)
    b = dense_rhs(cols, n, dtype=dtype)
    scorch.matmul(a, b)
    scorch.matmul(a, b)
    (plan,) = plans_of(a).values()
    return a, dense, b, plan


def test_declines_a_different_free_dimension():
    a, _, _, plan = _install()
    assert plan.run(a._raw_values, dense_rhs(64, 9), -1, False) is None


def test_declines_a_mismatched_contraction_extent():
    a, _, _, plan = _install()
    assert plan.run(a._raw_values, dense_rhs(63, 8), -1, False) is None


def test_declines_a_dtype_change():
    a, _, b, plan = _install()
    assert plan.run(a._raw_values, b.double(), -1, False) is None
    assert plan.run(a._raw_values.double(), b, -1, False) is None


def test_declines_a_non_contiguous_operand():
    a, _, _, plan = _install(n=8)
    transposed = dense_rhs(8, 64).T
    assert not transposed.is_contiguous()
    assert plan.run(a._raw_values, transposed, -1, False) is None


def test_declines_a_values_array_of_the_wrong_length():
    a, _, b, plan = _install()
    assert plan.run(a._raw_values[:-1], b, -1, False) is None


def test_declines_a_rank_mismatch():
    a, _, _, plan = _install()
    assert plan.run(a._raw_values, torch.rand(64), -1, False) is None
    assert plan.run(a._raw_values, torch.rand(64, 8, 1), -1, False) is None


def test_a_plan_that_only_ever_declines_is_withdrawn():
    """A plan that cannot serve a call site must stop charging it.

    A decline is not free: the lookup hits, ``run`` crosses into C++, screens, and
    returns nothing, and the ordinary path runs anyway -- 0.2-0.8 us on the small
    cells of both hosts. A call site that always passes a right-hand operand a plan
    cannot serve (here a transposed view, which has the planned shape and dtype but
    is not contiguous) would pay that forever. After
    ``MAX_FRUITLESS_DECLINES`` declines with nothing served, the plan is withdrawn and
    the product returns to costing exactly what it did before plans existed -- a plain
    dict miss.
    """
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    transposed = b.T.contiguous().T
    assert not transposed.is_contiguous(), "the operand under test must not be servable"
    expected = torch.matmul(dense.double(), b.double())

    installed_at = None
    withdrawn_at = None
    for call in range(1, 2 * MAX_FRUITLESS_DECLINES + 4):
        out = as_tensor(scorch.matmul(a, transposed))
        # Every call, planned or not, declined or not, returns the same right answer.
        torch.testing.assert_close(out.double(), expected, atol=1e-3, rtol=1e-3)
        if plans_of(a) and installed_at is None:
            installed_at = call
        if refused_of(a) and withdrawn_at is None:
            withdrawn_at = call

    assert installed_at == 2, f"plan installed on call {installed_at}, expected the 2nd"
    assert withdrawn_at is not None, "a plan that never served was never withdrawn"
    assert plans_of(a) == {}, "the withdrawn plan is still held"
    assert len(refused_of(a)) == 1
    # Withdrawn on the call after the threshold's worth of declines: installed on 2,
    # so declines run 2..9 and the 10th call sees the count reach 8.
    assert withdrawn_at == installed_at + MAX_FRUITLESS_DECLINES


def test_a_plan_that_serves_is_never_withdrawn():
    """The withdrawal must not touch a plan that is earning its keep, however many
    calls it also declines: an operand multiplied by both a contiguous B and a
    transposed view of the same shape keeps its plan for the calls it can serve."""
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    transposed = b.T.contiguous().T
    expected = torch.matmul(dense.double(), b.double())
    for _ in range(3 * MAX_FRUITLESS_DECLINES):
        for operand in (b, transposed):
            out = as_tensor(scorch.matmul(a, operand))
            torch.testing.assert_close(out.double(), expected, atol=1e-3, rtol=1e-3)
    (plan,) = plans_of(a).values()
    assert plan.served > MAX_FRUITLESS_DECLINES
    assert refused_of(a) == set(), "a plan that serves was withdrawn anyway"


def test_a_withdrawn_product_stops_costing_the_native_call():
    """After withdrawal the probe must miss outright, not find something to ask.

    This is what makes the withdrawal worth anything: the value is stored as ``None``,
    and ``ops.matmul`` already tests the looked-up value, so a refused product costs a
    dict miss and nothing else. Pinned by the plan's own counter -- if the withdrawn
    plan were still being consulted, ``served`` aside, the object would still be
    reachable from the operand.
    """
    a, _ = random_csr(64, 64, 0.05)
    transposed = dense_rhs(64, 8).T.contiguous().T
    for _ in range(2 * MAX_FRUITLESS_DECLINES + 2):
        scorch.matmul(a, transposed)
    (key,) = refused_of(a)
    held = a.__dict__[plan_mod.PLANS_ATTR]
    assert held[key] is None, "a withdrawn product must hold None, not a plan"


def test_declining_still_produces_the_right_answer_through_matmul():
    """A transposed right-hand operand is the realistic decline; it must work."""
    a, dense, _, plan = _install(n=8)
    transposed = dense_rhs(8, 64).T
    before = plan.served
    out = as_tensor(scorch.matmul(a, transposed))
    assert plan.served == before, "the plan served a call it should have declined"
    torch.testing.assert_close(
        out,
        torch.matmul(dense.double(), transposed.double()).float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_declines_after_the_index_array_is_written():
    """A structural write must not be served from a plan that predates it."""
    a, _, b, plan = _install()
    positions, coordinates = a._native_mode_indices()[1]
    coordinates[0] = coordinates[0]  # a no-op write still bumps the version counter
    assert plan.run(a._raw_values, b, -1, False) is None
    assert positions is not None


# --------------------------------------------------------------------------- #
# Invalidation
# --------------------------------------------------------------------------- #


def test_setting_state_drops_the_plan():
    """``_set_state`` is the funnel every in-place structural change goes through,
    so the drop is tested there directly -- no compiler needed -- as well as
    through the public mutators below."""
    a, _, _, _ = _install()
    assert plans_of(a)
    a._set_state(a._metadata, a._storage)
    assert plans_of(a) == {}


@pytest.mark.parametrize(
    "duplicate,description",
    [
        (copy.copy, "copy.copy"),
        (copy.deepcopy, "copy.deepcopy"),
        (lambda t: t.copy(), "STensor.copy"),
        (lambda t: pickle.loads(pickle.dumps(t)), "pickle round trip"),
    ],
)
def test_an_operand_with_a_plan_can_still_be_copied_and_pickled(duplicate, description):
    """Every one of these works on an operand that has never been multiplied, so
    every one has to keep working on one that has.

    A plan lives in the native extension and cannot be pickled, so before
    ``STensor.__getstate__`` dropped it, `deepcopy` and `pickle` raised ``TypeError:
    cannot pickle 'scorch_ops.SpmmCsrPlan' object`` on any operand that had been used
    twice -- a regression created by plans, on calls that have nothing to do with
    them. The duplicate must also carry no plan of its own (a plan is memoized work,
    not state) and must still compute the right product, and the original must keep
    the plan it had.
    """
    a, dense, b, _ = _install()
    assert plans_of(a), "nothing to test: the original has no plan"
    duplicated = duplicate(a)
    assert plans_of(duplicated) == {}, f"{description} carried the plan across"
    torch.testing.assert_close(
        as_tensor(scorch.matmul(duplicated, b)).double(),
        torch.matmul(dense.double(), b.double()),
        atol=1e-3,
        rtol=1e-3,
    )
    assert len(plans_of(a)) == 1, f"{description} disturbed the original's plan"
    torch.testing.assert_close(
        as_tensor(scorch.matmul(a, b)).double(),
        torch.matmul(dense.double(), b.double()),
        atol=1e-3,
        rtol=1e-3,
    )


def test_relayout_drops_the_plan_and_keeps_the_answer_right(generated_kernels):
    a, dense, b, _ = _install()
    assert plans_of(a)
    a.change_mode_order([1, 0])
    assert plans_of(a) == {}, "a relaid-out operand kept its plan"
    a.change_mode_order([0, 1])
    out = as_tensor(scorch.matmul(a, b))
    torch.testing.assert_close(
        out,
        torch.matmul(dense.double(), b.double()).float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_a_transposed_operand_fails_the_same_way_and_is_never_planned():
    """A plan may not change what a call that already fails does.

    ``change_mode_order([1, 0])`` is a transpose: the (32, 48) operand becomes a
    (48, 32) one whose stored indices are still the original row-major CSR.
    ``scorch.matmul`` raises on such an operand **on both hosts and on the tree
    before plans existed** -- prebuilt resolution keys on the format string and the
    rank, never on the mode order, so it picks the row-major kernel and that
    kernel's ABI guard rejects the shapes it is handed (a pre-existing defect,
    unrelated to plans: it fails closed, but with a kernel-level ``RuntimeError``
    rather than a scorch-level structured error).

    What is under test is that plans neither paper over that failure nor attach
    anything to the operand: the exception must be the same one, every time.
    """
    _, dense = random_csr(32, 48, 0.1)
    a = csr_stensor(dense)
    a.change_mode_order([1, 0])
    assert a.storage.index.mode_order == [1, 0]
    assert tuple(a.shape) == (48, 32)
    b = dense_rhs(32, 8)

    with pytest.raises(Exception) as without_plans:
        matmul_without_plans(a, b)
    for _ in range(3):
        with pytest.raises(Exception) as with_plans:
            scorch.matmul(a, b)
        assert type(with_plans.value) is type(without_plans.value)
        assert str(with_plans.value) == str(without_plans.value)
    assert plans_of(a) == {}


# --------------------------------------------------------------------------- #
# Every kernel kind a plan can hold, against the symbol it stands in for
# --------------------------------------------------------------------------- #


def legacy_args(dense, b):
    csr = dense.to_sparse_csr()
    positions = csr.crow_indices().to(torch.int32)
    coordinates = csr.col_indices().to(torch.int32)
    values = csr.values()
    rows, cols = dense.shape
    n = b.shape[1]
    return (
        [rows, n],
        [rows, cols],
        [[], [positions, coordinates]],
        values,
        [cols, n],
        [[], []],
        b.reshape(-1),
    )


@pytest.mark.parametrize("nthreads", [-1, 1, 2])
def test_plan_matches_the_v2_symbol_bitwise(nthreads):
    generator = torch.Generator().manual_seed(3)
    dense = (torch.rand((96, 96), generator=generator) < 0.1).float()
    dense *= torch.rand((96, 96), generator=generator)
    b = dense_rhs(96, 12)
    args = legacy_args(dense, b)
    legacy = scorch_ops.spmm_csr_float_v2(
        *args,
        tile_size=min(256, max(b.shape[1], 1)),
        nthreads_override=nthreads,
        atparallel=False,
    ).storage.value.reshape(96, 12)
    plan = scorch_ops.make_spmm_csr_plan("v2", args[1], args[2], args[3], 12)
    torch.testing.assert_close(
        plan.run(args[3], b, nthreads, False),
        legacy,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("panel", [0, 16, 64])
def test_plan_matches_the_tilej_symbol_bitwise(panel):
    generator = torch.Generator().manual_seed(4)
    dense = (torch.rand((128, 256), generator=generator) < 0.08).float()
    dense *= torch.rand((128, 256), generator=generator)
    b = dense_rhs(256, 24)
    args = legacy_args(dense, b)
    legacy = scorch_ops.spmm_csr_float_tilej(
        *args,
        Jc=panel,
        nthreads_override=2,
    ).storage.value.reshape(128, 24)
    plan = scorch_ops.make_spmm_csr_plan(
        "tilej",
        args[1],
        args[2],
        args[3],
        24,
        0,
        panel,
    )
    torch.testing.assert_close(
        plan.run(args[3], b, 2, False),
        legacy,
        atol=0,
        rtol=0,
    )
    assert plan.kind == "tilej"


@pytest.mark.parametrize("panels", [(0, 0), (16, 64), (8, 32)])
def test_plan_matches_the_tileijk_symbol_bitwise(panels):
    free, contraction = panels
    generator = torch.Generator().manual_seed(5)
    dense = (torch.rand((96, 192), generator=generator) < 0.08).float()
    dense *= torch.rand((96, 192), generator=generator)
    b = dense_rhs(192, 32)
    args = legacy_args(dense, b)
    legacy = scorch_ops.spmm_csr_float_tileijk(
        *args,
        Nc=free,
        Jc=contraction,
        nthreads_override=2,
    ).storage.value.reshape(96, 32)
    plan = scorch_ops.make_spmm_csr_plan(
        "tileijk",
        args[1],
        args[2],
        args[3],
        32,
        free,
        contraction,
    )
    torch.testing.assert_close(
        plan.run(args[3], b, 2, False),
        legacy,
        atol=0,
        rtol=0,
    )
    assert plan.kind == "tileijk"


def test_plan_matches_the_f64_symbol_bitwise():
    generator = torch.Generator().manual_seed(6)
    dense = (torch.rand((80, 80), generator=generator) < 0.1).double()
    dense *= torch.rand((80, 80), generator=generator).double()
    b = dense_rhs(80, 16, dtype=torch.float64)
    args = legacy_args(dense, b)
    legacy = scorch_ops.prebuilt_spmm_csr_f64(*args).storage.value.reshape(80, 16)
    plan = scorch_ops.make_spmm_csr_plan("reference", args[1], args[2], args[3], 16)
    torch.testing.assert_close(
        plan.run(args[3], b, -1, False),
        legacy,
        atol=0,
        rtol=0,
    )


def test_factory_refuses_what_it_cannot_serve():
    generator = torch.Generator().manual_seed(7)
    dense = (torch.rand((32, 32), generator=generator) < 0.2).float()
    args = legacy_args(dense, dense_rhs(32, 4))
    make = scorch_ops.make_spmm_csr_plan
    assert make("nonsense", args[1], args[2], args[3], 4) is None
    # float32 values cannot be served by the float64 reference kernel, and the
    # tiled kernels reject a zero extent rather than guess.
    assert make("reference", args[1], args[2], args[3], 4) is None
    assert make("tileijk", args[1], args[2], args[3], 0) is None
    assert make("v2", [32], args[2], args[3], 4) is None


def test_plan_narrows_int64_indices_once_and_stays_bitwise_exact():
    generator = torch.Generator().manual_seed(8)
    dense = (torch.rand((72, 72), generator=generator) < 0.1).float()
    dense *= torch.rand((72, 72), generator=generator)
    csr = dense.to_sparse_csr()
    positions = csr.crow_indices().to(torch.int64)
    coordinates = csr.col_indices().to(torch.int64)
    values = csr.values()
    b = dense_rhs(72, 8)
    mode_indices = [[], [positions, coordinates]]
    legacy = scorch_ops.spmm_csr_float_v2(
        [72, 8],
        [72, 72],
        mode_indices,
        values,
        [72, 8],
        [[], []],
        b.reshape(-1),
    ).storage.value.reshape(72, 8)
    plan = scorch_ops.make_spmm_csr_plan("v2", [72, 72], mode_indices, values, 8)
    assert plan is not None
    torch.testing.assert_close(plan.run(values, b, -1, False), legacy, atol=0, rtol=0)
    expected = torch.matmul(dense.double(), b.double()).float()
    torch.testing.assert_close(
        plan.run(values, b, -1, False),
        expected,
        atol=1e-3,
        rtol=1e-3,
    )


def test_plan_reports_its_shape():
    a, _, _, plan = _install(rows=48, cols=64, n=8, density=0.1)
    assert (plan.rows, plan.cols, plan.free_dim) == (48, 64, 8)
    assert plan.nnz == int(a._native_mode_indices()[1][1].numel())


# --------------------------------------------------------------------------- #
# A training-loop shape: repeated products through the public API
# --------------------------------------------------------------------------- #


def test_repeated_layer_stack_matches_the_unplanned_path():
    """The pattern plans exist for: one adjacency, several free dimensions, many
    iterations, values updated in between."""
    a, dense = random_csr(256, 256, 0.02, seed=11)
    widths = [64, 16, 4]
    features = [dense_rhs(256, n, seed=20 + n) for n in widths]

    expected = []
    for b in features:
        expected.append(as_tensor(matmul_without_plans(a, b)))

    got = [[] for _ in widths]
    for _ in range(4):
        for index, b in enumerate(features):
            got[index].append(as_tensor(scorch.matmul(a, b)))

    assert len(plans_of(a)) == len(widths)
    for index, outs in enumerate(got):
        for out in outs:
            torch.testing.assert_close(out, expected[index], atol=0, rtol=0)


def test_numpy_backed_operand_is_planned_correctly():
    """scipy/numpy-sourced indices arrive as int64 and are narrowed once."""
    rows = 128
    degree = 3
    rng = np.random.default_rng(0)
    columns = np.concatenate(
        [np.sort(rng.choice(rows, size=degree, replace=False)) for _ in range(rows)]
    ).astype(np.int64)
    positions = np.arange(rows + 1, dtype=np.int64) * degree
    values = rng.standard_normal(rows * degree).astype(np.float32)
    a = STensor.from_torch(
        torch.sparse_csr_tensor(
            torch.from_numpy(positions),
            torch.from_numpy(columns),
            torch.from_numpy(values),
            size=(rows, rows),
        )
    )
    b = dense_rhs(rows, 16)
    reference = as_tensor(matmul_without_plans(a, b))
    scorch.matmul(a, b)
    scorch.matmul(a, b)
    torch.testing.assert_close(
        as_tensor(scorch.matmul(a, b)),
        reference,
        atol=0,
        rtol=0,
    )


# --------------------------------------------------------------------------- #
# Autotune policy changes
# --------------------------------------------------------------------------- #


def test_changing_the_global_autotune_level_retires_plans():
    """A plan carries the selector's verdict, so a policy change must retire it."""
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    previous = scorch.get_autotune()
    try:
        for _ in range(3):
            scorch.matmul(a, b)
        (plan,) = plans_of(a).values()
        served = plan.served
        scorch.set_autotune("off")
        out = as_tensor(scorch.matmul(a, b))
        assert plan.served == served, "a retired plan still served a call"
        torch.testing.assert_close(
            out,
            torch.matmul(dense.double(), b.double()).float(),
            atol=1e-3,
            rtol=1e-3,
        )
    finally:
        scorch.set_autotune(previous)


def test_autotune_context_manager_is_never_served_by_an_outside_plan():
    a, dense = random_csr(64, 64, 0.05)
    b = dense_rhs(64, 8)
    for _ in range(3):
        scorch.matmul(a, b)
    (plan,) = plans_of(a).values()
    served = plan.served
    with scorch.autotune("off"):
        inside = as_tensor(scorch.matmul(a, b))
    assert plan.served == served, "a plan built outside the block served inside it"
    torch.testing.assert_close(
        inside,
        torch.matmul(dense.double(), b.double()).float(),
        atol=1e-3,
        rtol=1e-3,
    )
    # Leaving the block retires that generation too, so the next call rebuilds.
    for _ in range(3):
        scorch.matmul(a, b)
    torch.testing.assert_close(
        as_tensor(scorch.matmul(a, b)),
        torch.matmul(dense.double(), b.double()).float(),
        atol=1e-3,
        rtol=1e-3,
    )


def test_sparse_right_hand_operand_is_not_mistaken_for_a_dense_one():
    """``matmul(csr, dense_STensor)`` is a legitimate call shape and must not be
    mistaken for the planned one -- the installer once read the free dimension off
    a right-hand operand it had already rejected."""
    a, dense = random_csr(64, 64, 0.05)
    b_dense = dense_rhs(64, 8)
    b_wrapped = STensor.from_torch(b_dense)
    for _ in range(3):
        out = scorch.matmul(a, b_wrapped)
    torch.testing.assert_close(
        as_tensor(out),
        torch.matmul(dense.double(), b_dense.double()).float(),
        atol=1e-3,
        rtol=1e-3,
    )
    assert plans_of(a) == {}, "a wrapped right-hand operand was planned"


# --------------------------------------------------------------------------- #
# The tiling selector's verdict, carried into a plan
# --------------------------------------------------------------------------- #


def scattered_csr(rows, degree, seed):
    """A scattered high-degree matrix -- what the tiling gate is looking for."""
    generator = torch.Generator().manual_seed(seed)
    dense = torch.zeros(rows, rows)
    for row in range(rows):
        columns = torch.randperm(rows, generator=generator)[:degree]
        dense[row, columns] = torch.randn(degree, generator=generator)
    return csr_stensor(dense), dense


# rows/degree/free dim chosen against tiling's gate, not by taste: with the LLC
# forced to 128 KiB, B is 800*4*64 = 200 KiB (> C, so it thrashes) and the degree
# of 70 clears tiling._DEG_FLOOR of 64. A shape that misses either -- the earlier
# version of this test used degree 40 -- never dispatches a tiled kernel at all,
# so it cannot test that the plan follows one.
TILED_ROWS = 800
TILED_DEGREE = 70
TILED_N = 64
TILED_LLC = 131072


def test_a_tiled_verdict_is_carried_into_the_plan():
    """When the selector has chosen a tiled kernel for a shape, the plan must run
    that kernel with those panel widths -- not v2, and not a different width.

    The verdict is written into the memo directly rather than probed for: what is
    under test is that ``ops.matmul`` hands the plan the same kernel and parameters
    the ordinary path actually dispatched, which is independent of whether tiling
    wins on this machine's cache. The shape still has to pass the gate, or the
    ordinary path runs v2 and there is no verdict to carry.
    """
    from scorch import tiling

    a, dense = scattered_csr(TILED_ROWS, TILED_DEGREE, seed=21)
    b = dense_rhs(TILED_ROWS, TILED_N, seed=22)

    previous_llc = tiling._llc_bytes
    previous_level = scorch.get_autotune()
    try:
        tiling._llc_bytes = TILED_LLC  # force the gate open for this shape
        scorch.set_autotune("balanced")
        assert tiling.is_candidate(a, b), "the shape does not reach the selector"
        tiling._decision.clear()
        signature = tiling._signature(a, TILED_N)
        tiling._decision[(signature, "balanced")] = ("tilej", 64)

        reference = as_tensor(matmul_without_plans(a, b))
        for _ in range(3):
            planned = as_tensor(scorch.matmul(a, b))
        (plan,) = plans_of(a).values()
        assert plan.kind == "tilej", f"plan holds {plan.kind}, not the memoized winner"
        assert plan.served >= 1
        torch.testing.assert_close(planned, reference, atol=0, rtol=0)
        torch.testing.assert_close(
            planned.double(),
            torch.matmul(dense.double(), b.double()),
            atol=1e-3,
            rtol=1e-3,
        )
    finally:
        tiling._llc_bytes = previous_llc
        tiling._decision.clear()
        scorch.set_autotune(previous_level)


def test_a_memo_entry_the_gate_rejects_leaves_the_plan_on_v2():
    """A memo entry is not on its own an instruction to run a tiled kernel.

    The selector memoizes per (operand signature, level) and the entry outlives the
    conditions that produced it: the same operand at a smaller free dimension, or on
    a machine whose LLC swallows B, fails the O(1) gate and is served by v2 without
    the memo ever being consulted. An earlier version of this wiring read the memo
    from the installer, which sees every call, so a plan ran tile-j against an
    ordinary path running v2 -- bit-different output (4.8e-06 on a 400x400) from a
    path that is supposed to be indistinguishable.
    """
    from scorch import tiling

    a, dense = scattered_csr(TILED_ROWS, TILED_DEGREE, seed=25)
    narrow = 8  # B is now 25 KiB: under the forced LLC, so the gate says no
    b = dense_rhs(TILED_ROWS, narrow, seed=26)

    previous_llc = tiling._llc_bytes
    previous_level = scorch.get_autotune()
    try:
        tiling._llc_bytes = TILED_LLC
        scorch.set_autotune("balanced")
        assert not tiling.is_candidate(a, b), "the gate admitted the narrow shape"
        tiling._decision.clear()
        # A stale winner for exactly this operand and free dimension. The ordinary
        # path never looks at it; neither may the plan.
        signature = tiling._signature(a, narrow)
        tiling._decision[(signature, "balanced")] = ("tilej", 64)

        reference = as_tensor(matmul_without_plans(a, b))
        for _ in range(3):
            planned = as_tensor(scorch.matmul(a, b))
        (plan,) = plans_of(a).values()
        assert plan.kind == "v2", f"plan holds {plan.kind}; the gate rejected tiling"
        assert plan.served >= 1
        torch.testing.assert_close(planned, reference, atol=0, rtol=0)
        torch.testing.assert_close(
            planned.double(),
            torch.matmul(dense.double(), b.double()),
            atol=1e-3,
            rtol=1e-3,
        )
    finally:
        tiling._llc_bytes = previous_llc
        tiling._decision.clear()
        scorch.set_autotune(previous_level)


def test_a_v2_verdict_keeps_the_plan_on_v2():
    """The same shape with the selector's usual verdict stays on v2."""
    from scorch import tiling

    a, _ = random_csr(400, 400, 0.1, seed=23)
    b = dense_rhs(400, 32, seed=24)
    previous_llc = tiling._llc_bytes
    try:
        tiling._llc_bytes = 131072
        tiling._decision.clear()
        for _ in range(3):
            scorch.matmul(a, b)
        (plan,) = plans_of(a).values()
        assert plan.kind == "v2"
    finally:
        tiling._llc_bytes = previous_llc
        tiling._decision.clear()


@pytest.mark.parametrize(
    "rows,cols,n,description",
    [
        (8, 8, 0, "zero free dimension"),
        (0, 8, 4, "no rows"),
        (8, 0, 4, "zero contraction extent"),
        (1, 1, 1, "one of everything"),
    ],
)
def test_degenerate_shapes_agree_with_the_ordinary_path(rows, cols, n, description):
    """A plan must agree with the ordinary path about degenerate extents --
    including about whether they are an error at all, which is why the two paths
    are compared by outcome and not just by value."""
    dense = torch.zeros(rows, cols)
    if rows and cols:
        dense[0, 0] = 1.0
    a = csr_stensor(dense)
    b = torch.ones(cols, n)

    def outcome(planned):
        plan_mod.forget(a)
        plan_mod.set_enabled(planned)
        try:
            if planned:
                scorch.matmul(a, b)
                scorch.matmul(a, b)
            out = as_tensor(scorch.matmul(a, b))
            return ("value", tuple(out.shape), out.clone())
        except Exception as exc:
            return ("raise", type(exc).__name__, str(exc))
        finally:
            plan_mod.set_enabled(True)

    ordinary = outcome(False)
    planned = outcome(True)
    assert (
        ordinary[:2] == planned[:2]
    ), f"{description}: {ordinary[:2]} vs {planned[:2]}"
    if ordinary[0] == "value":
        torch.testing.assert_close(planned[2], ordinary[2], atol=0, rtol=0)
