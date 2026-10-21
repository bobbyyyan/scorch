"""``scorch.matmul`` of a matrix and a vector returns one type per result format.

A 2-D ``A`` times a 1-D ``x`` is computed by a prebuilt kernel when one matches (a CSR
matrix in default mode order, a dense vector, float32, float64, int32 or int64, and
``use_cache`` on), by ``torch.matmul`` when both operands are dense, and otherwise by
the generated matrix-vector kernel (``ops.spmv``).  The prebuilt branch returned a
``torch.Tensor`` for a dense result and the generated branch an ``STensor`` of format
``"d"``, so the type of ``matmul(A, x)`` followed the layout of ``A``, ``use_cache`` and
the value dtype.  The rule the product of two matrices already follows holds here now:
a dense result is a ``torch.Tensor``, a sparse result an ``STensor``.

A case that names the code it expects to compute the product checks that it did, with a
recording wrapper around ``ops._spmv_owned`` (the generated kernel's entry, which
``ops.spmv`` and ``matmul`` both call), so it cannot pass by taking the other branch.
Values are compared with PyTorch's product of the dense operands: float64 to 1e-12 of
each cell, float32 to 1e-5, integers exactly; a sparse result's level arrays are decoded
without the library's conversions (``level_array_support``).  Operands have rows and
vector entries that store nothing, at 5 rows and at 300.

Some cases omit ``format=``: the defect was in that call, and the default result format
of this product is the dense vector, fixed in ``ops.spmv`` and not inferred.
"""

import itertools

import pytest
import torch

import scorch
from scorch import STensor, ops
from tests.test_scorch.level_array_support import assert_matches, stored

MATRIX = ["ds", "ss", "sd", "oo", "dd"]
VECTOR = ["d", "s", "o"]
SHAPES = {"small": (5, 4), "large": (300, 40)}
DTYPES = [torch.float64, torch.float32, torch.int64, torch.int32]


def _operands(shape, dtype=torch.float64, seed=53):
    """``A`` and ``x`` as dense tensors of ``dtype``, and the reference ``A @ x``.

    Which entries are stored is fixed and the values are random.  Rows 1, 4, 7, ... of
    ``A`` store nothing, the other rows store one entry in three, and entries 2, 6,
    10, ... of ``x`` store nothing, so a row can store entries and still have a zero
    product (row 2 of the 5 by 4 matrix does).

    The reference of a floating dtype is the float64 product of the operands as stored
    in ``dtype``, so a float32 result is judged against a product that carries no
    float32 rounding of its own; an integer reference is the int64 product, in ``dtype``.
    """

    rows, cols = shape
    generator = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        a = 0.5 + torch.rand(shape, generator=generator, dtype=torch.float64)
        x = 0.5 + torch.rand((cols,), generator=generator, dtype=torch.float64)
    else:
        a = torch.randint(1, 9, shape, generator=generator).to(torch.float64)
        x = torch.randint(1, 9, (cols,), generator=generator).to(torch.float64)
    row = torch.arange(rows).reshape(-1, 1)
    col = torch.arange(cols).reshape(1, -1)
    a = (a * (((row + 2 * col) % 3 == 0) & (row % 3 != 1))).to(dtype)
    x = (x * (torch.arange(cols) % 4 != 2)).to(dtype)
    if dtype.is_floating_point:
        expected = a.to(torch.float64) @ x.to(torch.float64)
    else:
        expected = (a.to(torch.int64) @ x.to(torch.int64)).to(dtype)
    assert int((expected != 0).sum()) > 0
    assert int((expected == 0).sum()) > 0
    return a, x, expected


def _route(matrix_format, vector_format, use_cache=True):
    """Which code computes ``matmul(A, x)`` for these layouts and a dense result."""

    if matrix_format == "dd" and vector_format == "d":
        return "torch"
    if use_cache and matrix_format == "ds" and vector_format == "d":
        return "prebuilt"
    return "generated"


@pytest.fixture
def generated(monkeypatch):
    """The calls that reached the generated matrix-vector kernel, one entry each.

    It wraps ``ops._spmv_owned``, which both ``ops.spmv`` and ``matmul``'s
    matrix-vector branch call (``matmul`` calls it directly, under the call's own
    options holder), and records the result format each call asked for, its third
    argument.  The prebuilt and the ``torch.matmul`` branches return before it.
    """

    calls = []
    generic_spmv = ops._spmv_owned

    def spmv_owned(a, b, output_format, *args, **kwargs):
        calls.append(output_format)
        return generic_spmv(a, b, output_format, *args, **kwargs)

    monkeypatch.setattr(ops, "_spmv_owned", spmv_owned)
    return calls


@pytest.mark.parametrize("size", ["small", "large"])
@pytest.mark.parametrize("requested", [None, "d"], ids=["format omitted", "format d"])
@pytest.mark.parametrize(
    ("matrix_format", "vector_format"), list(itertools.product(MATRIX, VECTOR))
)
def test_a_dense_result_is_a_torch_tensor_whichever_code_computes_it(
    matrix_format, vector_format, requested, size, generated
):
    a, x, expected = _operands(SHAPES[size])
    keywords = {} if requested is None else {"format": requested}

    result = scorch.matmul(
        stored(a, matrix_format, "A"), stored(x, vector_format, "x"), **keywords
    )

    took_generated = _route(matrix_format, vector_format) == "generated"
    assert len(generated) == (1 if took_generated else 0)
    assert type(result) is torch.Tensor
    assert result.dtype is torch.float64
    assert_matches(result, expected, torch.float64)


@pytest.mark.parametrize("size", ["small", "large"])
@pytest.mark.parametrize(
    ("matrix_format", "vector_format", "result_format"),
    list(itertools.product(["ds", "oo"], ["d", "s"], ["s", "o"])),
)
def test_a_sparse_result_is_an_stensor(
    matrix_format, vector_format, result_format, size
):
    a, x, expected = _operands(SHAPES[size])

    result = scorch.matmul(
        stored(a, matrix_format, "A"),
        stored(x, vector_format, "x"),
        format=result_format,
    )

    assert type(result) is STensor
    wanted = scorch.parse_format(result_format)
    assert tuple(result.format.level_types) == tuple(wanted.level_types)
    assert result.dtype is torch.float64
    assert_matches(result, expected, torch.float64)


@pytest.mark.parametrize("size", ["small", "large"])
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda dtype: str(dtype).split(".")[-1])
def test_the_prebuilt_and_the_generated_kernel_agree_in_type_and_value(
    dtype, size, generated
):
    a, x, expected = _operands(SHAPES[size], dtype)

    prebuilt = scorch.matmul(stored(a, "ds", "A"), stored(x, "d", "x"), format="d")
    assert generated == [], "the call did not resolve to a prebuilt kernel"
    built = scorch.matmul(
        stored(a, "ds", "A"), stored(x, "d", "x"), format="d", use_cache=False
    )
    assert len(generated) == 1, "use_cache=False did not reach the generated kernel"

    for result in (prebuilt, built):
        assert type(result) is torch.Tensor
        assert result.dtype is dtype
        assert_matches(result, expected, dtype)


@pytest.mark.parametrize("size", ["small", "large"])
@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_a_torch_sparse_matrix_and_a_torch_vector_give_a_torch_tensor(
    layout, size, generated
):
    a, x, expected = _operands(SHAPES[size])
    sparse = a.to_sparse_coo() if layout == "coo" else a.to_sparse_csr()

    result = scorch.matmul(sparse, x)

    # A torch CSR operand is the prebuilt kernel's layout; a torch COO operand has no
    # prebuilt kernel for a vector and takes the generated one.
    assert len(generated) == (1 if layout == "coo" else 0)
    assert type(result) is torch.Tensor
    assert result.dtype is torch.float64
    assert_matches(result, expected, torch.float64)


def test_the_kernel_time_is_still_reported(generated):
    a, x, expected = _operands(SHAPES["large"])
    time_dict = {}

    result = scorch.matmul(stored(a, "oo", "A"), x, time_dict=time_dict)

    assert len(generated) == 1
    assert type(result) is torch.Tensor
    assert time_dict["eval_time"] >= 0.0
    assert_matches(result, expected, torch.float64)


def test_the_internal_entry_still_returns_the_stensor_it_built():
    a, x, expected = _operands(SHAPES["small"])

    result = ops.spmv(stored(a, "oo", "A"), stored(x, "d", "x"))

    assert type(result) is STensor
    assert result.format.is_dense()
    assert_matches(result, expected, torch.float64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_generated_kernel_s_product_has_the_einsums_derivative(generated, dtype):
    """The generated kernel's product is differentiable.  A keyword argument takes the
    product out of what autograd.py's product Functions cover, and the product is
    recorded with the einsum's derivative instead, for both operands, as the product of
    two matrices is (test_autograd_matmul.py).  Before step 3 its backward raised
    matmul_derivative_unsupported, and before that the dense result left as an STensor,
    which carried no autograd at all."""

    a, x, expected = _operands(SHAPES["small"], dtype)
    matrix = a.to_sparse_csr().requires_grad_()
    weights = x.clone().requires_grad_()

    out = scorch.matmul(matrix, weights, use_cache=False)

    assert len(generated) == 1
    assert type(out) is torch.Tensor
    assert_matches(out.detach(), expected, dtype)
    upstream = torch.rand(out.shape, generator=torch.Generator().manual_seed(5))
    upstream = upstream.to(torch.float64)
    (out * upstream.to(dtype)).sum().backward()

    # The float64 reference, on the operands as stored in dtype.
    a64, x64 = a.to(torch.float64), x.to(torch.float64)
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    close = dict(rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(weights.grad.double(), a64.t() @ upstream, **close)
    assert matrix.grad.layout is torch.sparse_csr
    torch.testing.assert_close(
        matrix.grad.to_dense().double(),
        torch.outer(upstream, x64) * (a64 != 0),
        **close,
    )


@pytest.mark.parametrize("size", ["small", "large"])
def test_a_compiled_function_applies_its_later_operations_to_the_product(size):
    """``scorch.compile`` applies the traced operations after the product to what
    ``matmul`` returns, so they need the ``torch.Tensor``.  Which kernel computes the
    product inside the compiled function is not checked here."""

    a, x, _ = _operands(SHAPES[size])
    # Odd rows negated: the product has entries of both signs, so relu changes it.
    rows = torch.arange(a.shape[0]).reshape(-1, 1)
    a = a * (1.0 - 2.0 * (rows % 2))
    product = a @ x
    assert bool((product > 0).any()) and bool((product < 0).any())

    def fn(matrix, vector):
        return torch.relu(scorch.matmul(matrix, vector))

    result = scorch.compile(fn)(stored(a, "oo", "A"), x)

    assert type(result) is torch.Tensor
    assert_matches(result, torch.relu(product), torch.float64)
