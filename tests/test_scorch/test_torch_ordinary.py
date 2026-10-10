"""A torch CSR operand's unplanned product, made by the C entry point.

``scorch.matmul`` on a torch sparse CSR tensor that has no plan used to go to the Python
function, which wraps both operands in STensors to make one call to the drop-in SpMM. The C
entry point now makes that call itself (``csrc/planned_matmul.h``,
``planned_matmul_torch_ordinary``) when it is exactly the call the Python function would make,
and hands everything else to it. These tests pin both halves: the products it makes are
bit-identical to the Python function's, and every product it must not make -- one the tiling
selector may want, a malformed operand, a B the kernel cannot read as given -- still reaches the
Python function, which reports what it always reported.
"""

import warnings

import pytest
import torch

import scorch
import scorch_ops
from scorch import ops
from scorch import plan as plan_mod
from scorch import storage as storage_mod
from scorch import tiling
from scorch.exceptions import TensorIndexError, TensorValidationError
from scorch.plan import plans_of

warnings.filterwarnings("ignore", message=".*Sparse.*")

pytestmark = pytest.mark.skipif(
    not hasattr(scorch_ops, "set_planned_matmul_ordinary"),
    reason="this build has no C path for unplanned torch operands",
)


def served():
    return scorch_ops.planned_matmul_ordinary_served()


def arrays(rows, cols, density, dtype=torch.float32, index=torch.int64, seed=0):
    """(crow, col, values) of a random CSR matrix with an empty middle row."""
    generator = torch.Generator().manual_seed(seed)
    mask = torch.rand((rows, cols), generator=generator) < density
    dense = (torch.rand((rows, cols), generator=generator) * mask).to(dtype)
    if rows > 2:
        dense[rows // 2] = 0
    a = dense.to_sparse_csr()
    return a.crow_indices().to(index), a.col_indices().to(index), a.values()


def rhs(cols, k, dtype=torch.float32, seed=1):
    return torch.rand((cols, k), generator=torch.Generator().manual_seed(seed)).to(
        dtype
    )


def csr(crow, col, values, rows, cols):
    return torch.sparse_csr_tensor(crow, col, values, size=(rows, cols))


def python_product(crow, col, values, rows, cols, b):
    """The Python function's product over copies of the same arrays."""
    return ops._matmul_python(csr(crow.clone(), col.clone(), values, rows, cols), b)


@pytest.fixture
def plans_on():
    enabled = plan_mod.enabled()
    plan_mod.set_enabled(True)
    yield
    plan_mod.set_enabled(enabled)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("index", [torch.int64, torch.int32])
@pytest.mark.parametrize(
    "rows, cols, density, k",
    [
        (1, 1, 1.0, 1),
        (7, 5, 0.4, 3),
        (64, 80, 0.0, 4),
        (300, 257, 0.05, 1),
        (1000, 900, 0.01, 33),
    ],
)
def test_the_product_it_makes_is_the_python_functions(
    plans_on, dtype, index, rows, cols, density, k
):
    crow, col, values = arrays(rows, cols, density, dtype, index)
    b = rhs(cols, k, dtype)
    before = served()
    got = scorch.matmul(csr(crow.clone(), col.clone(), values, rows, cols), b)
    assert served() == before + 1
    want = python_product(crow, col, values, rows, cols, b)
    assert got.dtype == want.dtype and got.shape == want.shape and got.is_contiguous()
    assert torch.equal(got, want)
    dense = torch.sparse_csr_tensor(crow, col, values, size=(rows, cols)).to_dense()
    torch.testing.assert_close(got, dense @ b, atol=1e-4, rtol=1e-4)


def test_new_arrays_every_call_are_made_here_every_call(plans_on):
    crow, col, values = arrays(200, 150, 0.05)
    b = rhs(150, 8)
    before = served()
    for _ in range(5):
        scorch.matmul(csr(crow.clone(), col.clone(), values, 200, 150), b)
    assert served() == before + 5


def test_a_persistent_operand_is_still_planned_on_its_second_call(plans_on):
    crow, col, values = arrays(200, 150, 0.05)
    a = csr(crow, col, values, 200, 150)
    b = rhs(150, 8)
    before = served()
    outs = [scorch.matmul(a, b) for _ in range(4)]
    # The first call is a first sighting, made here; the second installs a plan, which is the
    # Python function's; the rest are the plan's.
    assert served() == before + 1
    assert len(plans_of(a)) == 1
    for out in outs[1:]:
        assert torch.equal(out, outs[0])


def test_with_plans_off_every_call_is_the_python_functions():
    crow, col, values = arrays(100, 100, 0.05)
    b = rhs(100, 4)
    enabled = plan_mod.enabled()
    plan_mod.set_enabled(False)
    try:
        before = served()
        scorch.matmul(csr(crow.clone(), col.clone(), values, 100, 100), b)
        assert served() == before
    finally:
        plan_mod.set_enabled(enabled)


@pytest.mark.parametrize(
    "case", ["noncontiguous_b", "b_dtype", "b_vector", "b_grad", "a_grad", "keyword"]
)
def test_products_it_does_not_make_are_the_python_functions(plans_on, case):
    crow, col, values = arrays(60, 50, 0.1)
    a = csr(crow.clone(), col.clone(), values, 60, 50)
    b = rhs(50, 6)
    kwargs = {}
    if case == "noncontiguous_b":
        b = rhs(6, 50).t()
    elif case == "b_dtype":
        b = b.double()
    elif case == "b_vector":
        b = b[:, 0].contiguous()
    elif case == "b_grad":
        b.requires_grad_(True)
    elif case == "a_grad":
        leaf = values.clone().requires_grad_(True)
        a = csr(crow.clone(), col.clone(), leaf, 60, 50)
    elif case == "keyword":
        kwargs = {"use_cache": True}
    before = served()
    if case == "b_dtype":
        # The Python function does not make a mixed-dtype product either.
        with pytest.raises(TensorValidationError, match="same value dtype"):
            scorch.matmul(a, b)
        assert served() == before
        return
    got = scorch.matmul(a, b, **kwargs)
    want = a.detach().to_dense() @ b.detach()
    torch.testing.assert_close(got.detach(), want, atol=1e-4, rtol=1e-4)
    if case in ("b_grad", "a_grad"):
        # autograd.py records the product; its forward computes it with grad mode off, and
        # that inner product is an ordinary one, which may be made here.
        assert got.requires_grad and got.grad_fn is not None
        got.sum().backward()
        grad = b.grad if case == "b_grad" else leaf.grad
        assert (
            grad is not None and grad.shape == (b if case == "b_grad" else leaf).shape
        )
    else:
        assert served() == before


def _bad(case):
    crow = torch.tensor([0, 2, 3, 5], dtype=torch.int64)
    col = torch.tensor([0, 2, 1, 0, 3], dtype=torch.int64)
    if case == "unsorted":
        col = torch.tensor([2, 0, 1, 0, 3], dtype=torch.int64)
    elif case == "out_of_range":
        col = torch.tensor([0, 2, 1, 0, 4], dtype=torch.int64)
    elif case == "negative":
        col = torch.tensor([0, 2, -1, 0, 3], dtype=torch.int64)
    elif case == "decreasing":
        crow = torch.tensor([0, 3, 2, 5], dtype=torch.int64)
    return crow, col, torch.rand(5)


@pytest.mark.parametrize("index", [torch.int64, torch.int32])
@pytest.mark.parametrize("case", ["unsorted", "out_of_range", "negative", "decreasing"])
def test_a_malformed_operand_is_reported_by_the_python_function(plans_on, index, case):
    crow, col, values = _bad(case)
    crow, col = crow.to(index), col.to(index)
    b = rhs(4, 3)
    with torch.sparse.check_sparse_tensor_invariants(False):
        with pytest.raises(Exception) as python_raised:
            ops._matmul_python(csr(crow.clone(), col.clone(), values, 3, 4), b)
        before = served()
        with pytest.raises(type(python_raised.value)) as raised:
            scorch.matmul(csr(crow.clone(), col.clone(), values, 3, 4), b)
    assert served() == before
    assert str(raised.value) == str(python_raised.value)
    assert isinstance(raised.value, TensorIndexError)


@pytest.fixture
def gate():
    """Install a different tiling gate for float32; restore the real one afterwards."""

    def install(fn):
        scorch_ops.set_planned_matmul_ordinary(fn, True, True, storage_mod._WRAP_GRAIN)

    yield install
    install(tiling.shape_is_candidate)


@pytest.mark.parametrize("verdict", ["candidate", "raises", "not_a_bool"])
def test_a_float32_shape_the_gate_does_not_decline_is_the_python_functions(
    plans_on, gate, verdict
):
    asked = []

    def fn(J, N, nnz):
        asked.append((J, N, nnz))
        if verdict == "raises":
            raise RuntimeError("gate failed")
        return True if verdict == "candidate" else 0

    gate(fn)
    crow, col, values = arrays(60, 50, 0.1)
    b = rhs(50, 6)
    before = served()
    got = scorch.matmul(csr(crow.clone(), col.clone(), values, 60, 50), b)
    assert served() == before
    assert asked == [(50, 6, values.numel())]
    assert torch.equal(got, python_product(crow, col, values, 60, 50, b))
    # float64 never enters the selector, so the gate is not asked.
    before = served()
    scorch.matmul(csr(crow.clone(), col.clone(), values.double(), 60, 50), b.double())
    assert served() == before + 1 and len(asked) == 1


def test_the_gate_is_the_one_the_python_function_asks(plans_on):
    """shape_is_candidate and is_candidate agree, at every level, on shapes either side of it."""
    crow, col, values = arrays(64, 64, 0.1)
    a = scorch.STensor.from_torch(csr(crow, col, values, 64, 64))
    llc = tiling.query_llc()
    for J, N in ((64, 8), (1 << 20, 64), (llc // 4, 16), (llc, 256)):
        for level in ("off", "analytic", "balanced", "max", "learned"):
            fake_a = type("A", (), {})()
            fake_a.shape = (64, J)
            fake_a.storage = a.storage
            b = torch.empty(J, N)
            with scorch.autotune(level):
                assert tiling.shape_is_candidate(
                    J, N, values.numel()
                ) == tiling.is_candidate(fake_a, b), (J, N, level)


def test_writes_to_the_arrays_between_calls_are_seen(plans_on):
    rows, cols = 50, 40
    crow, col, values = arrays(rows, cols, 0.1)
    b = rhs(cols, 5)
    first = scorch.matmul(csr(crow, col, values, rows, cols), b)
    # The same arrays, written in place through the caller's own array: each row's columns less
    # its first, which keeps every row sorted and in range.
    starts = crow[:-1].repeat_interleave(crow[1:] - crow[:-1])
    with torch.no_grad():
        col.sub_(col[starts])
    second = scorch.matmul(csr(crow, col, values, rows, cols), b)
    want = torch.sparse_csr_tensor(crow, col, values, size=(rows, cols)).to_dense() @ b
    torch.testing.assert_close(second, want)
    assert not torch.equal(first, second)


def test_writes_through_an_operands_accessor_are_seen(plans_on):
    rows, cols = 50, 40
    crow, col, values = arrays(rows, cols, 0.1)
    b = rhs(cols, 5)
    a = csr(crow, col, values, rows, cols)
    scorch.matmul(a, b)  # `a` is now a witness of the arrays it was seen with
    starts = crow[:-1].repeat_interleave(crow[1:] - crow[:-1])
    shifted = col - col[starts]
    a.col_indices().copy_(shifted)
    got = scorch.matmul(csr(crow, col, values, rows, cols), b)
    want = (
        torch.sparse_csr_tensor(crow, shifted, values, size=(rows, cols)).to_dense() @ b
    )
    torch.testing.assert_close(got, want)


def test_inference_tensors_are_served(plans_on):
    with torch.inference_mode():
        crow, col, values = arrays(80, 70, 0.1)
        b = rhs(70, 4)
        before = served()
        got = scorch.matmul(csr(crow, col, values, 80, 70), b)
    assert served() == before + 1
    torch.testing.assert_close(got, csr(crow, col, values, 80, 70).to_dense() @ b)
