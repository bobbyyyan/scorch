"""Two torch CSR operands' product, returned as the torch sparse CSR tensor torch returns.

``scorch.matmul`` on two torch sparse CSR tensors used to wrap both in STensors and return the
product as an STensor, a type no caller holding two torch tensors asked for, after 234 Python
calls on a small product. The product is now made by one native function,
``spgemm_csr_torch_result`` -- called by the C entry point (``csrc/planned_matmul.h``,
``planned_matmul_torch_spgemm``) and by the Python function (``ops._torch_csr_spgemm``) alike --
and returned as a torch sparse CSR tensor that satisfies torch's own invariants.

The kernel does not need a row's coordinates in order, and sums a coordinate repeated within a
row, so an operand torch itself produced unsorted (``torch.sparse.mm``'s own CSR x CSR result)
is read as it is. That is a claim, so it is tested here on rows reversed, on rows with repeated
coordinates and on torch.sparse.mm's own output, each through both routes, with the C entry
point's counter showing it made the product without handing it to the Python function.
"""

import gc
import warnings

import numpy as np
import pytest
import torch

import scorch
import scorch_ops
from scorch import ops
from scorch.exceptions import ScorchNotImplementedError

warnings.filterwarnings("ignore", message=".*Sparse.*")

pytestmark = pytest.mark.skipif(
    not hasattr(scorch_ops, "set_planned_matmul_spgemm"),
    reason="this build has no torch CSR x CSR product",
)

DTYPES = [torch.float32, torch.float64, torch.int32, torch.int64]


def served():
    return scorch_ops.planned_matmul_spgemm_served()


def random_csr(rows, cols, density, dtype, index=torch.int64, seed=0, min_per_row=0):
    """A torch CSR tensor with sorted, distinct coordinates in every row; values nonzero."""
    rng = np.random.default_rng(seed)
    mask = rng.random((rows, cols)) < density
    for i in range(rows if cols else 0):
        if min_per_row:
            mask[i, rng.choice(cols, min(min_per_row, cols), replace=False)] = True
    if dtype.is_floating_point:
        values = rng.uniform(-1.0, 1.0, (rows, cols))
        values[values == 0] = 0.5
    else:
        values = rng.integers(1, 4, (rows, cols)) * rng.choice([-1, 1], (rows, cols))
    dense = torch.from_numpy(np.where(mask, values, 0)).to(dtype)
    a = dense.to_sparse_csr()
    return torch.sparse_csr_tensor(
        a.crow_indices().to(index), a.col_indices().to(index), a.values(), size=(rows, cols)
    )


def reversed_rows(a):
    """The same matrix with every row's entries stored in reverse order."""
    crow, col, values = a.crow_indices(), a.col_indices().clone(), a.values().clone()
    for i in range(a.shape[0]):
        lo, hi = int(crow[i]), int(crow[i + 1])
        col[lo:hi] = col[lo:hi].flip(0)
        values[lo:hi] = values[lo:hi].flip(0)
    return torch.sparse_csr_tensor(crow.clone(), col, values, size=a.shape)


def repeated_coordinates(a):
    """The same matrix with every row of two or more entries storing its first coordinate a
    second time, the value split between the two, and the row's order reversed: a CSR tensor
    whose to_dense() equals a's, as torch sums repeated entries."""
    crow, col, values = a.crow_indices(), a.col_indices(), a.values()
    out_crow, out_col, out_val = [0], [], []
    for i in range(a.shape[0]):
        lo, hi = int(crow[i]), int(crow[i + 1])
        c, v = col[lo:hi].tolist(), values[lo:hi].tolist()
        if len(c) >= 2:
            if a.dtype.is_floating_point:
                half = v[0] / 2
                v = [half] + v[1:] + [v[0] - half]
            else:
                v = [1] + v[1:] + [v[0] - 1]
            c = c + [c[0]]
            c, v = c[::-1], v[::-1]
        out_col += c
        out_val += v
        out_crow.append(len(out_col))
    index = crow.dtype
    return torch.sparse_csr_tensor(
        torch.tensor(out_crow, dtype=index),
        torch.tensor(out_col, dtype=index),
        torch.tensor(out_val, dtype=a.dtype),
        size=a.shape,
    )


def reference(a, b):
    """float64 a @ b from the operands' own arrays, summing repeated entries."""

    def dense(x):
        out = np.zeros(x.shape, dtype=np.float64)
        crow = x.crow_indices().numpy()
        rows = np.repeat(np.arange(x.shape[0]), np.diff(crow))
        np.add.at(out, (rows, x.col_indices().numpy()), x.values().double().numpy())
        return out

    return dense(a) @ dense(b)


def check(result, a, b):
    """The result is a valid torch CSR tensor of the provisional index dtype, equal to the
    float64 reference."""
    assert type(result) is torch.Tensor
    assert result.layout is torch.sparse_csr
    assert result.dtype is a.dtype
    assert result.shape == (a.shape[0], b.shape[1])
    index = torch.int64 if ops._SPGEMM_INT64_INDICES else torch.int32
    assert result.crow_indices().dtype is index
    assert result.col_indices().dtype is index
    # torch's own invariants: sorted, distinct coordinates in every row, among others.
    torch._validate_sparse_csr_tensor_args(
        result.crow_indices(), result.col_indices(), result.values(), result.shape
    )
    want = reference(a, b)
    got = result.to_dense().double().numpy()
    if a.dtype.is_floating_point:
        tol = 1e-12 if a.dtype is torch.float64 else 1e-5
        # initial=0.0: an empty product has nothing to compare.
        error = np.abs(got - want).max(initial=0.0)
        assert error <= tol * max(1.0, np.abs(want).max(initial=0.0))
    else:
        assert np.array_equal(got, want)


ROUTES = {
    "c_entry": lambda a, b: ops.matmul(a, b),
    "python": lambda a, b: ops._matmul_python(a, b),
}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("a_index", [torch.int64, torch.int32])
@pytest.mark.parametrize("b_index", [torch.int64, torch.int32])
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_product_is_torch_csr(dtype, a_index, b_index, route):
    a = random_csr(37, 23, 0.15, dtype, a_index, seed=1)
    b = random_csr(23, 41, 0.15, dtype, b_index, seed=2)
    before = served()
    check(ROUTES[route](a, b), a, b)
    assert served() - before == (1 if route == "c_entry" else 0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "rows, inner, cols, density",
    [(1, 1, 1, 1.0), (5, 0, 4, 0.0), (0, 6, 3, 0.5), (9, 7, 0, 0.5), (16, 16, 16, 0.0)],
)
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_empty_extents_and_empty_operands(dtype, rows, inner, cols, density, route):
    a = random_csr(rows, inner, density, dtype, seed=3)
    b = random_csr(inner, cols, density, dtype, seed=4)
    check(ROUTES[route](a, b), a, b)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("operand", ["reversed", "repeated"])
@pytest.mark.parametrize("which", ["a", "b", "both"])
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_order_free_operands_need_no_sorted_copy(dtype, operand, which, route):
    """A row's coordinates out of order, or repeated, are read as they are: the product is
    right, the C entry point makes it itself, and the operands' arrays are untouched."""
    make = reversed_rows if operand == "reversed" else repeated_coordinates
    a0 = random_csr(30, 20, 0.2, dtype, seed=5, min_per_row=2)
    b0 = random_csr(20, 25, 0.2, dtype, seed=6, min_per_row=2)
    a = make(a0) if which in ("a", "both") else a0
    b = make(b0) if which in ("b", "both") else b0
    with pytest.raises(RuntimeError):  # the operand really is out of order
        x = a if which in ("a", "both") else b
        torch._validate_sparse_csr_tensor_args(
            x.crow_indices(), x.col_indices(), x.values(), x.shape
        )
    held = [
        (t.data_ptr(), t.clone())
        for x in (a, b)
        for t in (x.crow_indices(), x.col_indices(), x.values())
    ]
    before = served()
    result = ROUTES[route](a, b)
    check(result, a, b)
    check(result, a0, b0)
    assert served() - before == (1 if route == "c_entry" else 0)
    now = [t for x in (a, b) for t in (x.crow_indices(), x.col_indices(), x.values())]
    for (ptr, copy), t in zip(held, now):
        assert t.data_ptr() == ptr
        assert torch.equal(t, copy)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_torch_sparse_mm_output_is_read_as_it_is(dtype, route):
    """torch.sparse.mm's own CSR x CSR result leaves rows unsorted; it is an operand like any
    other, here on both sides and chained."""
    s = random_csr(60, 60, 0.08, dtype, seed=7, min_per_row=2)
    try:
        t = torch.sparse.mm(s, s)
    except RuntimeError as error:
        # A torch built without MKL (the M5's) multiplies CSR x CSR only in COO.
        if "without MKL" not in str(error):
            raise
        pytest.skip("this torch has no CSR x CSR product: " + str(error)[:80])
    with pytest.raises(RuntimeError):  # unsorted rows, which this test needs
        torch._validate_sparse_csr_tensor_args(
            t.crow_indices(), t.col_indices(), t.values(), t.shape
        )
    before = served()
    check(ROUTES[route](t, s), t, s)
    check(ROUTES[route](s, t), s, t)
    check(ROUTES[route](t, t), t, t)
    assert served() - before == (3 if route == "c_entry" else 0)


def test_both_routes_return_the_same_tensor():
    a = random_csr(50, 40, 0.1, torch.float64, seed=8)
    b = random_csr(40, 30, 0.1, torch.float64, seed=9)
    c, p = ops.matmul(a, b), ops._matmul_python(a, b)
    assert torch.equal(c.crow_indices(), p.crow_indices())
    assert torch.equal(c.col_indices(), p.col_indices())
    assert torch.equal(c.values(), p.values())


def test_result_owns_its_arrays():
    """The result outlives its operands, and two results share no storage."""
    a = random_csr(40, 40, 0.2, torch.float64, seed=10)
    b = random_csr(40, 40, 0.2, torch.float64, seed=11)
    want = reference(a, b)
    first, second = ops.matmul(a, b), ops.matmul(a, b)
    del a, b
    gc.collect()
    for x in (first, second):
        assert np.abs(x.to_dense().numpy() - want).max() <= 1e-12 * np.abs(want).max()
    for f, s in zip(
        (first.crow_indices(), first.col_indices(), first.values()),
        (second.crow_indices(), second.col_indices(), second.values()),
    ):
        assert f.untyped_storage().data_ptr() != s.untyped_storage().data_ptr()
    second.values().zero_()
    assert np.abs(first.to_dense().numpy() - want).max() <= 1e-12 * np.abs(want).max()


def test_mismatched_inner_extent_is_the_ordinary_paths_error():
    a = random_csr(5, 4, 0.5, torch.float64, seed=12)
    b = random_csr(6, 3, 0.5, torch.float64, seed=13)
    before = served()
    with pytest.raises(Exception) as from_entry:
        ops.matmul(a, b)
    with pytest.raises(Exception) as from_python:
        ops._matmul_python(a, b)
    assert type(from_entry.value) is type(from_python.value)
    assert served() == before


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("index", [torch.int64, torch.int32])
@pytest.mark.parametrize("which", ["a", "b", "both"])
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_gradient_call_is_recorded_and_refuses_with_a_code(dtype, index, which, route):
    """A product a gradient is wanted through is autograd.py's: the forward is right, and
    asking for the gradient raises scorch's coded refusal rather than torch's uncoded one.

    The result is a torch CSR tensor, which has no view: autograd.py hands it to its
    Function held and returns a new tensor object over its arrays (draft 1c; 84357 found
    the forward raising NotImplementedError for aten::view on SparseCsrCPU)."""
    a = random_csr(20, 15, 0.2, dtype, index=index, seed=14)
    b = random_csr(15, 10, 0.2, dtype, index=index, seed=15)
    if which in ("a", "both"):
        a.requires_grad_(True)
    if which in ("b", "both"):
        b.requires_grad_(True)
    result = ROUTES[route](a, b)
    check(result.detach(), a.detach(), b.detach())
    assert result.grad_fn is not None
    assert result is not a and result is not b
    with pytest.raises(ScorchNotImplementedError) as refused:
        result.to_dense().sum().backward()
    assert refused.value.code == "matmul_derivative_unsupported"


def test_a_forced_schedule_is_not_served_by_the_c_entry():
    """Under schedule_force the C entry point hands the call to the Python function, which
    honours the schedule; the torch CSR x CSR route inherits that check."""
    a = random_csr(12, 10, 0.3, torch.float64, seed=16)
    b = random_csr(10, 8, 0.3, torch.float64, seed=17)
    schedule = scorch.Schedule(loop_order=("i", "j", "k"))
    before = served()
    with scorch.schedule_force(schedule):
        ops.matmul(a, b)
    assert served() == before


def test_serve_flags_follow_the_registry():
    from scorch.prebuilt_kernels import csr_csr_symbol

    for dtype, suffix in (
        (torch.float32, "f32"),
        (torch.float64, "f64"),
        (torch.int32, "i32"),
        (torch.int64, "i64"),
    ):
        assert ops._SPGEMM_SERVES[dtype] == (
            csr_csr_symbol(dtype) == f"prebuilt_spmspm_csr_{suffix}"
        )


# Keywords (patch A, recut): what each of matmul's keywords does to the type of the product.


def keyword_operands(dtype=torch.float64, index=torch.int64, operand="sorted", seed=20):
    a = random_csr(30, 25, 0.15, dtype, index=index, seed=seed, min_per_row=1)
    b = random_csr(25, 20, 0.15, dtype, index=index, seed=seed + 1, min_per_row=1)
    if operand == "reversed":
        a = reversed_rows(a)
    elif operand == "repeated":
        a = repeated_coordinates(a)
    return a, b


def check_csc(result, a, b):
    """The result is a valid torch CSC tensor of the provisional index dtype, equal to the
    float64 reference: the product stored column by column."""
    assert type(result) is torch.Tensor
    assert result.layout is torch.sparse_csc
    assert result.dtype is a.dtype
    assert result.shape == (a.shape[0], b.shape[1])
    index = torch.int64 if ops._SPGEMM_INT64_INDICES else torch.int32
    assert result.ccol_indices().dtype is index
    assert result.row_indices().dtype is index
    torch._validate_sparse_csc_tensor_args(
        result.ccol_indices(), result.row_indices(), result.values(), result.shape
    )
    check(result.to_sparse_csr(), a, b)


def no_schedule_options():
    from scorch.compiler.compile_options import CompileOptions

    options = CompileOptions.from_environment()
    assert options.requested_schedule is None
    return options


def no_schedule_context():
    from scorch.compiler.compilation_context import CompilationContext

    options = no_schedule_options()
    return {"_compile_options": options, "_compilation_context": CompilationContext(options)}


# Each keyword that leaves torch's type, with a value that is not its default (a default is
# dropped at the entry, and the call is then the plain one).
TORCH_TYPE_CALLS = {
    "use_cache": lambda: {"use_cache": False},
    "output_mode_order": lambda: {"output_mode_order": [1, 0]},
    "_compile_options": lambda: {"_compile_options": no_schedule_options()},
    "_compilation_context": no_schedule_context,
    "use_cache+output_mode_order": lambda: {"use_cache": False, "output_mode_order": [1, 0]},
}


def test_every_matmul_keyword_has_a_result_type_for_torch_csr_operands():
    """Every keyword matmul takes is placed: it asks for scorch's form, leaves torch's type, or
    is out= (the result itself). A keyword added to matmul fails here until it is placed."""
    stensor, torch_type = ops._STENSOR_RESULT_KEYWORDS, ops._TORCH_RESULT_KEYWORDS
    assert not stensor & torch_type
    assert stensor | torch_type | {"out"} == ops._MATMUL_KEYWORDS
    assert ops._SPGEMM_SERVED_KEYWORDS <= torch_type
    assert {name for call in TORCH_TYPE_CALLS.values() for name in call()} == torch_type - {
        "time_dict"
    }


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_time_dict_is_the_same_product_timed(dtype, route):
    """time_dict= is served by the function the plain call takes: the same arrays, bit for
    bit, and the native call's time under eval_time."""
    a, b = keyword_operands(dtype)
    plain = ops.matmul(a, b)
    times = {}
    entry = ops.matmul if route == "c_entry" else ops._matmul_python
    timed = entry(a, b, time_dict=times)
    check(timed, a, b)
    assert torch.equal(timed.crow_indices(), plain.crow_indices())
    assert torch.equal(timed.col_indices(), plain.col_indices())
    assert torch.equal(timed.values(), plain.values())
    assert isinstance(times["eval_time"], float) and times["eval_time"] >= 0.0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("call", sorted(TORCH_TYPE_CALLS))
@pytest.mark.parametrize("index", [torch.int64, torch.int32])
def test_a_keyword_without_a_format_or_schedule_keeps_torch_type(dtype, call, index):
    """use_cache=False, output_mode_order=, _compile_options= and _compilation_context=
    without a schedule: the product of the route each chooses, in torch's type -- torch CSR,
    or torch CSC for output_mode_order=[1, 0] -- with the plain call's index dtype, equal to
    the float64 reference. Before the recut each returned an STensor, the plain call torch
    CSR (RELAB 17894307)."""
    a, b = keyword_operands(dtype, index=index)
    kwargs = TORCH_TYPE_CALLS[call]()
    result = ops.matmul(a, b, **kwargs)
    if kwargs.get("output_mode_order") == [1, 0]:
        check_csc(result, a, b)
    else:
        check(result, a, b)


@pytest.mark.parametrize("operand", ["reversed", "repeated"])
@pytest.mark.parametrize("call", sorted(TORCH_TYPE_CALLS))
def test_order_free_operands_with_a_keyword(operand, call):
    """An operand torch accepts with a row out of order, or a coordinate repeated, through
    each route a keyword chooses: in torch's type, equal to the float64 reference (which sums
    repeated entries, as torch's to_dense does)."""
    a, b = keyword_operands(torch.float64, operand=operand)
    kwargs = TORCH_TYPE_CALLS[call]()
    result = ops.matmul(a, b, **kwargs)
    if kwargs.get("output_mode_order") == [1, 0]:
        check_csc(result, a, b)
    else:
        check(result, a, b)


def test_identity_output_mode_order_is_the_plain_call():
    a, b = keyword_operands()
    plain, ordered = ops.matmul(a, b), ops.matmul(a, b, output_mode_order=[0, 1])
    check(ordered, a, b)
    assert torch.equal(ordered.crow_indices(), plain.crow_indices())
    assert torch.equal(ordered.col_indices(), plain.col_indices())
    assert torch.equal(ordered.values(), plain.values())


@pytest.mark.parametrize("fmt", ["ds", "ss", "dd"])
def test_a_result_format_asks_for_scorchs_form(fmt):
    """format= names scorch's form: an STensor in that format, as on every other route, or
    for a dense format the dense torch tensor matmul returns for one; the same with
    output_format=, read as format=."""
    from scorch.format import parse_format
    from scorch.stensor import STensor

    a, b = keyword_operands()
    want = reference(a, b)
    for kwargs in ({"format": fmt}, {"output_format": fmt}):
        result = ops.matmul(a, b, **kwargs)
        if fmt == "dd":
            assert type(result) is torch.Tensor and result.layout is torch.strided
            got = result.double().numpy()
        else:
            assert type(result) is STensor and result.format == parse_format(fmt)
            got = result.to_torch(False).double().numpy()
        assert np.abs(got - want).max() <= 1e-12 * np.abs(want).max()


def test_a_schedule_asks_for_scorchs_form():
    """A schedule, as schedule= or forced by the context, and with a keyword that would leave
    torch's type: an STensor, the generated kernel's under that schedule."""
    from scorch.stensor import STensor

    a, b = keyword_operands()
    want = reference(a, b)
    schedule = scorch.Schedule(loop_order=("i", "j", "k"))
    results = [ops.matmul(a, b, schedule=schedule)]
    with scorch.schedule_force(schedule):
        results += [ops.matmul(a, b), ops.matmul(a, b, time_dict={})]
    for result in results:
        assert type(result) is STensor
        got = result.to_torch(False).double().numpy()
        assert np.abs(got - want).max() <= 1e-12 * np.abs(want).max()


def test_out_refuses_the_sparse_product_by_code():
    """out= takes a dense result; this product is sparse, so the call is refused by code and
    out is left as it was (before the recut, torch's uncoded copy_ error)."""
    from scorch.exceptions import KeywordArgumentError

    a, b = keyword_operands()
    out = torch.full((a.shape[0], b.shape[1]), 7.0, dtype=torch.float64)
    with pytest.raises(KeywordArgumentError) as refused:
        ops.matmul(a, b, out=out)
    assert refused.value.code == "out_requires_dense_result"
    assert bool((out == 7.0).all())


def malformed(kind):
    """A torch CSR operand torch builds without checking it (as by default), whose arrays
    break one rule the native function checks."""
    a, _ = keyword_operands()
    crow, col, values = a.crow_indices().clone(), a.col_indices().clone(), a.values()
    if kind == "column_out_of_range":
        col[0] = a.shape[1]
    elif kind == "negative_column":
        col[0] = -1
    else:  # "first_position_not_zero"
        crow[0] = 1
    return torch.sparse_csr_tensor(crow, col, values, size=a.shape)


@pytest.mark.parametrize(
    "kind", ["column_out_of_range", "negative_column", "first_position_not_zero"]
)
@pytest.mark.parametrize("route", sorted(ROUTES))
def test_a_malformed_operand_is_refused_by_code(kind, route):
    """The native function refuses a malformed operand with uncoded text; the call is then
    made on the ordinary path, which refuses it by code, as before patch A."""
    from scorch.exceptions import ScorchError, defect_code

    a = malformed(kind)
    _, b = keyword_operands()
    with pytest.raises(ScorchError) as refused:
        ROUTES[route](a, b)
    assert defect_code(refused.value) is not None
