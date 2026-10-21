"""Gradients of products whose sampled derivative stores nothing.

The gradient of a sparse operand ``X_k`` samples a product at ``X_k``'s stored
coordinates (``einsum_grad._Program.gradient``) and gathers it back into ``X_k``'s
storage order (``stensor_grad._gather``); forward mode gathers a tangent program the
same way into the result's storage order.  When the two operands' stored patterns do
not meet, or an operand or the upstream gradient stores nothing, that sample stores
nothing.  The gather then indexed an empty tensor and raised an uncoded IndexError, at
every result rank -- a full contraction such as ``ij,ij->`` or ``v @ v`` hit it most
often, since two sparse vectors drawn at random often share no entry.  The gradient
there is zero, on the operand's own stored entries.

Every case below is checked against torch on the dense tensors the operands hold, in
float64: the value, each operand's gradient at its stored entries (exactly, where
torch's is exactly zero), and where asked the second derivative and the forward-mode
tangent.  A call the compiler refuses for its formats is refused with a code, and
refused the same way when the patterns overlap: an empty sample never changes whether
a call is refused.
"""

from __future__ import annotations

import pytest
import torch
import torch.autograd.forward_ad as fwAD

import scorch
from scorch import stensor_grad
from scorch.exceptions import ScorchError, defect_code
from scorch.stensor import STensor

F32, F64 = torch.float32, torch.float64
AUTOGRAD_VALUES = stensor_grad.AUTOGRAD_VALUES
TOLERANCE = {F64: 1e-10, F32: 1e-5}  # rtol = atol; float32 sums of a few O(1) terms

# The operand formats of the drop-in scan's gradient keys (frame_dropin.SCAN_KINDS
# without torch's own layouts): (format, mode order); "csr32" and "csr64" are torch
# CSR tensors wrapped by STensor.from_csr with index arrays of that width.
FORMATS = {
    1: [("s", None)],
    2: [
        ("ds", None),
        ("sd", None),
        ("ss", None),
        ("oo", None),
        ("ds", (1, 0)),
        ("csr32", None),
        ("csr64", None),
    ],
    3: [("dds", None), ("dss", None), ("sss", None), ("ooo", None), ("dss", (2, 0, 1))],
}
# One format per rank for the cases that also run in float32 and with a dense operand.
FIRST = {1: FORMATS[1][0], 2: FORMATS[2][0], 3: FORMATS[3][1]}
SHAPES = {1: (6,), 2: (4, 5), 3: (3, 4, 2)}

# (name, logical shape, result format or None).  "same" names the operands' own
# format for a sparse result of their rank.
PROGRAMS = [
    ("i,i->", SHAPES[1], None),
    ("v@v", SHAPES[1], None),
    ("i,i->i", SHAPES[1], "same"),
    ("ij,ij->", SHAPES[2], None),
    ("ij,ij->ij", SHAPES[2], "same"),
    ("ij,ij->i", (1, 5), "s"),  # a rank-1 result of size 1
    ("ijk,ijk->", SHAPES[3], None),
]

# Stored positions (row-major linear indices) of the two operands, by name.
PATTERNS = {
    "disjoint": (lambda p: p % 2 == 0, lambda p: p % 2 == 1),
    "overlapping": (lambda p: p % 2 == 0, lambda p: p % 3 == 0),
    "left_empty": (lambda p: False, lambda p: p % 2 == 1),
    "both_empty": (lambda p: False, lambda p: False),
}

# A rank-2 matmul whose operands both store entries that never meet: A (4 x 5) stores
# only even columns and B (5 x 3) only odd rows, so no term A[i, k] B[k, j] has two
# stored factors.  A's gradient samples (upstream @ B^T) at A's entries, in columns
# where B stores no row; B's samples (A^T @ upstream) at B's, in rows where A stores
# no column.  "overlapping" stores B's rows 0 and 3, and row 0 meets A's column 0.
# A format whose inner level is dense ("sd") also stores the zeros of every row it
# keeps, so there the samples are not empty; the comparison with torch holds anyway.
MATMUL = ("A@B", ((4, 5), (5, 3)), None)
MATMUL_PATTERNS = {
    "disjoint": (lambda p: p % 5 % 2 == 0, lambda p: p // 3 % 2 == 1),
    "overlapping": (lambda p: p % 5 % 2 == 0, lambda p: p // 3 % 3 == 0),
}


def _dense(shape, stored, dtype):
    """Values 0.5 .. 1.5 at the positions ``stored`` picks, zero elsewhere."""

    n = 1
    for extent in shape:
        n *= extent
    p = torch.arange(n, dtype=F64)
    values = 0.5 + torch.remainder(p * 0.618, 1.0)
    mask = torch.tensor([bool(stored(int(i))) for i in range(n)])
    return (values * mask).reshape(shape).to(dtype)


def _template(dense, fmt, order):
    """An STensor of ``dense`` in ``fmt`` (and ``order``), without history."""

    with torch.no_grad():
        if fmt in ("csr32", "csr64"):
            csr = dense.to_sparse_csr()
            width = torch.int32 if fmt == "csr32" else torch.int64
            return scorch.from_csr(
                torch.sparse_csr_tensor(
                    csr.crow_indices().to(width),
                    csr.col_indices().to(width),
                    csr.values().clone(),
                    size=dense.shape,
                )
            )
        tensor = scorch.from_torch(dense.detach().clone())
        if order is not None:
            tensor.change_mode_order(list(order))
        if any(level != "d" for level in fmt):
            tensor.to_sparse(fmt)
    return tensor


def _over(template, values):
    """``values`` (and their history) in ``template``'s structure, as an STensor."""

    return STensor.from_components(
        tuple(template.shape),
        template.format,
        [list(arrays) for arrays in template._native_mode_indices()],
        values,
        mode_order=list(template.mode_order),
    )


def _logical(template, values):
    """The dense tensor of ``values`` at ``template``'s coordinates, differentiable."""

    keys = stensor_grad._Structure.of(template).keys()
    shape = tuple(template.logical_shape)
    flat = torch.zeros(int(torch.tensor(shape).prod()), dtype=values.dtype)
    return flat.index_add(0, keys, values.to(flat.dtype)).reshape(shape)


def _dense_of(result):
    return result.to_torch(in_place=False) if type(result) is STensor else result


def _result_format(program, fmt, order):
    name, shape, result = program
    if result != "same":
        return result
    return "ds" if fmt in ("csr32", "csr64") else fmt


def _call(program, fmt, order, a, b):
    name = program[0]
    if name in ("v@v", "A@B"):
        return scorch.matmul(a, b)
    result = _result_format(program, fmt, order)
    kwargs = {} if result is None else {"format": result}
    return scorch.einsum(name, a, b, **kwargs)


def _torch_call(program, a, b):
    name = program[0]
    return a @ b if name in ("v@v", "A@B") else torch.einsum(name, a, b)


def _operands(program, fmt, order, pattern, dtype, dense_b=False, asked_b=True):
    """The two operands' templates and values; ``b``'s values ask for a gradient
    only when ``asked_b``."""

    if program is MATMUL:
        (shape_a, shape_b), (left, right) = program[1], MATMUL_PATTERNS[pattern]
    else:
        (shape_a, shape_b), (left, right) = (program[1],) * 2, PATTERNS[pattern]
    ta = _template(_dense(shape_a, left, dtype), fmt, order)
    va = ta._raw_values.clone().requires_grad_()
    if dense_b:
        b = _dense(shape_b, right, dtype).requires_grad_(asked_b)
        return ta, va, None, b
    tb = _template(_dense(shape_b, right, dtype), fmt, order)
    vb = tb._raw_values.clone().requires_grad_(asked_b)
    return ta, va, tb, vb


def _weights(shape):
    n = 1
    for extent in shape:
        n *= extent
    return (1.0 + torch.arange(n, dtype=F64) / 7.0).reshape(shape)


def _refusal_or_none(program, fmt, order, pattern, dtype, dense_b, asked_b=True):
    ta, va, tb, vb = _operands(program, fmt, order, pattern, dtype, dense_b, asked_b)
    a, b = _over(ta, va), (vb if tb is None else _over(tb, vb))
    try:
        _call(program, fmt, order, a, b)
    except ScorchError as error:
        return defect_code(error) or "uncoded"
    return None


def _compare(actual, reference, dtype):
    """``actual`` against the float64 ``reference``: exactly when the reference is
    exactly zero (patterns that do not meet), else at the dtype's tolerance."""

    tol = 0.0 if not bool(reference.any()) else TOLERANCE[dtype]
    torch.testing.assert_close(
        actual.detach().to(F64), reference.detach(), rtol=tol, atol=tol
    )


def _check_gradients(program, fmt, order, pattern, dtype, dense_b=False, asked_b=True):
    ta, va, tb, vb = _operands(program, fmt, order, pattern, dtype, dense_b, asked_b)
    a, b = _over(ta, va), (vb if tb is None else _over(tb, vb))
    try:
        out = _call(program, fmt, order, a, b)
    except ScorchError as error:
        code = defect_code(error)
        assert code is not None, f"uncoded refusal: {error!r}"
        # A refusal for the formats, never for the empty sample: the overlapping
        # operands are refused the same way.
        again = _refusal_or_none(
            program, fmt, order, "overlapping", dtype, dense_b, asked_b
        )
        assert again == code
        return "refused"
    dense = _dense_of(out)
    # the reference: torch on the same values, in float64
    va64 = va.detach().to(F64).requires_grad_()
    vb64 = vb.detach().to(F64).requires_grad_(asked_b)
    a64 = _logical(ta, va64)
    b64 = vb64 if tb is None else _logical(tb, vb64)
    ref = _torch_call(program, a64, b64)
    _compare(dense, ref, dtype)
    w = _weights(tuple(ref.shape))
    leaves, leaves64 = ((va, vb), (va64, vb64)) if asked_b else ((va,), (va64,))
    got = torch.autograd.grad((dense * w.to(dense.dtype)).sum(), leaves)
    want = torch.autograd.grad((ref * w).sum(), leaves64)
    for g, r in zip(got, want):
        assert g is not None and g.shape == r.shape
        _compare(g, r, dtype)
    return "computed"


def _cases(full_formats):
    for program in PROGRAMS:
        rank = len(program[1])
        formats = FORMATS[rank] if full_formats else [FIRST[rank]]
        for fmt, order in formats:
            yield program, fmt, order


def _id(program, fmt, order):
    return f"{program[0]} {fmt}{'' if order is None else ''.join(map(str, order))}"


CORE = [
    pytest.param(
        program,
        fmt,
        order,
        pattern,
        dtype,
        id=f"{_id(program, fmt, order)} {pattern} {str(dtype)[6:]}",
    )
    for program, fmt, order in _cases(full_formats=False)
    for pattern in PATTERNS
    for dtype in (F64, F32)
    if dtype == F64 or pattern in ("disjoint", "overlapping")
]


@pytest.mark.parametrize("program,fmt,order,pattern,dtype", CORE)
def test_two_sparse_operands(program, fmt, order, pattern, dtype):
    _check_gradients(program, fmt, order, pattern, dtype)


EVERY_FORMAT = [
    pytest.param(program, fmt, order, id=_id(program, fmt, order))
    for program, fmt, order in _cases(full_formats=True)
    if (fmt, order) != FIRST[len(program[1])]
]


@pytest.mark.parametrize("program,fmt,order", EVERY_FORMAT)
def test_disjoint_patterns_in_every_format(program, fmt, order):
    _check_gradients(program, fmt, order, "disjoint", F64)


DENSE_SECOND = [
    pytest.param(
        program, fmt, order, pattern, id=f"{_id(program, fmt, order)} {pattern}"
    )
    for program, fmt, order in _cases(full_formats=False)
    if program[2] is None  # only these give a sparse product with a dense operand
    for pattern in ("overlapping", "left_empty")
]


@pytest.mark.parametrize("program,fmt,order,pattern", DENSE_SECOND)
def test_one_sparse_operand_and_a_dense_one(program, fmt, order, pattern):
    _check_gradients(program, fmt, order, pattern, F64, dense_b=True)


ONLY_FIRST = [
    pytest.param(
        program, fmt, order, pattern, id=f"{_id(program, fmt, order)} {pattern}"
    )
    for program, fmt, order in _cases(full_formats=False)
    for pattern in ("disjoint", "overlapping")
]


@pytest.mark.parametrize("program,fmt,order,pattern", ONLY_FIRST)
def test_only_the_first_operand_asks_for_a_gradient(program, fmt, order, pattern):
    """Both operands sparse, and only the first's values ask for a gradient: that
    gradient alone, against torch's."""

    _check_gradients(program, fmt, order, pattern, F64, asked_b=False)


MATMUL_CASES = [
    pytest.param(
        fmt,
        order,
        pattern,
        dtype,
        id=f"{_id(MATMUL, fmt, order)} {pattern} {str(dtype)[6:]}",
    )
    for fmt, order in FORMATS[2]
    for pattern in MATMUL_PATTERNS
    for dtype in (F64, F32)
    if dtype == F64 or (fmt, order) == FIRST[2]
]


@pytest.mark.parametrize("fmt,order,pattern,dtype", MATMUL_CASES)
def test_a_matmul_whose_operands_never_meet(fmt, order, pattern, dtype):
    _check_gradients(MATMUL, fmt, order, pattern, dtype)


SUMS = [
    pytest.param(rank, pattern, dtype, id=f"rank {rank} {pattern} {str(dtype)[6:]}")
    for rank in (1, 2, 3)
    for pattern in ("negated", "disjoint", "left_empty", "both_empty")
    for dtype in (F64, F32)
    if dtype == F64 or pattern in ("negated", "disjoint")
]


@pytest.mark.parametrize("rank,pattern,dtype", SUMS)
def test_a_sum_that_cancels_or_whose_patterns_do_not_meet(rank, pattern, dtype):
    """``a + (-a)`` ("negated": the same entries, values that cancel exactly) and
    ``a + b``, through ``stensor_grad._SummedValues``: the value and the gradients
    against torch's, exactly zero where torch's are.  The sum stores the union of
    the operands' coordinates (asserted: a sum that dropped a cancelled entry would
    store fewer), so its backward gathers each operand's entries from a target that
    stores all of them."""

    fmt, order = FIRST[rank]
    left, right = PATTERNS["disjoint" if pattern == "negated" else pattern]
    ta = _template(_dense(SHAPES[rank], left, dtype), fmt, order)
    va = ta._raw_values.clone().requires_grad_()
    va64 = va.detach().to(F64).requires_grad_()
    if pattern == "negated":
        a, b = _over(ta, va), _over(ta, -va)
        ref = _logical(ta, va64) + _logical(ta, -va64)
        leaves, leaves64 = (va,), (va64,)
    else:
        tb = _template(_dense(SHAPES[rank], right, dtype), fmt, order)
        vb = tb._raw_values.clone().requires_grad_()
        vb64 = vb.detach().to(F64).requires_grad_()
        a, b = _over(ta, va), _over(tb, vb)
        ref = _logical(ta, va64) + _logical(tb, vb64)
        leaves, leaves64 = (va, vb), (va64, vb64)
    total = a + b
    keys = torch.cat([stensor_grad._Structure.of(t).keys() for t in (a, b)])
    assert total._raw_values.numel() == torch.unique(keys).numel()
    dense = _dense_of(total)
    _compare(dense, ref, dtype)
    w = _weights(tuple(ref.shape))
    got = torch.autograd.grad((dense * w.to(dense.dtype)).sum(), leaves)
    want = torch.autograd.grad((ref * w).sum(), leaves64)
    for g, r in zip(got, want):
        assert g is not None and g.shape == r.shape
        _compare(g, r, dtype)


SECOND = [
    pytest.param(PROGRAMS[k], pattern, id=f"{PROGRAMS[k][0]} {pattern}")
    for k in (0, 2)
    for pattern in ("disjoint", "overlapping")
]


@pytest.mark.parametrize("program,pattern", SECOND)
def test_second_derivative_of_a_disjoint_pair_is_zero_not_missing(program, pattern):
    """create_graph=True: the first gradient stays recorded through the sample, so the
    second derivative is a tensor (zeros where the patterns do not meet), not None and
    not an error, and equals torch's."""

    fmt, order = FIRST[1]
    ta, va, tb, vb = _operands(program, fmt, order, pattern, F64)
    out = _dense_of(_call(program, fmt, order, _over(ta, va), _over(tb, vb)))
    w = _weights(tuple(out.shape))
    (first,) = torch.autograd.grad((out * w).sum(), (va,), create_graph=True)
    ramp = torch.arange(1.0, first.numel() + 1, dtype=F64)
    (second,) = torch.autograd.grad((first * ramp).sum(), (vb,))
    va64 = va.detach().clone().requires_grad_()
    vb64 = vb.detach().clone().requires_grad_()
    ref = _torch_call(program, _logical(ta, va64), _logical(tb, vb64))
    (rfirst,) = torch.autograd.grad((ref * w).sum(), (va64,), create_graph=True)
    (rsecond,) = torch.autograd.grad((rfirst * ramp).sum(), (vb64,))
    assert second is not None
    torch.testing.assert_close(first.detach(), rfirst.detach(), rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(second, rsecond, rtol=1e-10, atol=1e-10)
    if pattern == "disjoint":
        assert not bool(second.any())


@pytest.mark.parametrize("program,pattern", SECOND)
def test_gradgradcheck(program, pattern):
    """torch.autograd.gradgradcheck in float64: the second derivatives autograd
    records, against finite differences of the first gradient (zero where the
    patterns do not meet)."""

    fmt, order = FIRST[1]
    ta, va, tb, vb = _operands(program, fmt, order, pattern, F64)

    def call(x, y):
        return _dense_of(_call(program, fmt, order, _over(ta, x), _over(tb, y)))

    assert torch.autograd.gradgradcheck(call, (va, vb))


@pytest.mark.parametrize("program,pattern", SECOND)
def test_forward_mode_of_a_disjoint_pair(program, pattern):
    """The tangent program of a disjoint pair stores nothing; the tangent of the
    result's stored values is then zero (gathered into the result's storage order)."""

    fmt, order = FIRST[1]
    ta, va, tb, vb = _operands(program, fmt, order, pattern, F64)
    tangent = torch.linspace(1.0, 2.0, va.numel(), dtype=F64)
    with fwAD.dual_level():
        # the operands carry history, so an STensor result may carry the tangent (0.2)
        dual = fwAD.make_dual(va, tangent)
        out = _call(program, fmt, order, _over(ta, dual), _over(tb, vb))
        values = out
        if type(out) is STensor:
            values = out.__dict__.get(AUTOGRAD_VALUES, out)
        got = fwAD.unpack_dual(values).tangent
        got_dense = got if type(out) is not STensor else _logical(out.detach(), got)
    a_dot = _logical(ta, tangent)
    want = _torch_call(program, a_dot, _logical(tb, vb.detach()))
    torch.testing.assert_close(got_dense.to(F64), want, rtol=1e-10, atol=1e-10)


# Conversions gather the gradient (and scatter the tangent, stensor_grad._scatter)
# between two structures; a target that keeps no entry of its source is the same case.
CONVERSIONS = [
    ("empty ds to_dense", "ds", lambda t: t.to_dense(in_place=False)),
    ("empty ds change_mode_order", "ds", lambda t: t.change_mode_order([1, 0])),
    ("empty oo change_mode_order", "oo", lambda t: t.change_mode_order([1, 0])),
    ("empty ss copy", "ss", lambda t: t.copy()),
    ("empty ds coalesce", "ds", lambda t: t.coalesce()),
    ("zero dd to_sparse", "dd", lambda t: t.to_sparse("ds")),
]


@pytest.mark.parametrize(
    "fmt,convert", [pytest.param(f, c, id=n) for n, f, c in CONVERSIONS]
)
def test_a_conversion_that_keeps_no_entry(fmt, convert):
    """Backward and forward mode through a conversion whose target stores nothing (an
    empty source, or a dense source of zeros made sparse), against torch's own
    to_sparse semantics: the gradient at an entry the target does not store is zero."""

    template = _template(torch.zeros(4, 5, dtype=F64), fmt, None)
    v = template._raw_values.clone().requires_grad_()
    result = convert(_over(template, v))
    dense = _dense_of(result)
    w = _weights((4, 5))
    (got,) = torch.autograd.grad((dense * w).sum(), (v,), allow_unused=True)
    got = torch.zeros_like(v) if got is None else got
    if fmt == "dd":
        # torch: zeros(4, 5).to_sparse() stores nothing, and its gradient is zero there
        x = torch.zeros(4, 5, dtype=F64, requires_grad=True)
        (want,) = torch.autograd.grad((x.to_sparse().to_dense() * w).sum(), (x,))
        torch.testing.assert_close(got, want.reshape(-1), rtol=0, atol=0)
    else:
        assert got.numel() == 0
    with fwAD.dual_level():
        dual = fwAD.make_dual(v, torch.ones_like(v))
        result = convert(_over(template, dual))
        values = result
        if type(result) is STensor:
            values = result.__dict__.get(AUTOGRAD_VALUES)
        if values is not None:
            tangent = fwAD.unpack_dual(values).tangent
            if tangent is not None and fmt != "dd":
                assert tangent.numel() == 0 or not bool(tangent.any())
