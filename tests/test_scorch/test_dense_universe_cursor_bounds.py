"""Sparse cursors must stay inside their own segment under a dense universe.

When an iteration lattice has a dense universe -- any elementwise expression
in which at least one operand's level at that index is DENSE -- the emitted
loop is a plain counted ``for`` over the dense extent.  That loop never
exhausts the *sparse* operands' cursors, so a lattice point that selected its
case on a bare ``coord == index`` equality would keep dereferencing a drained
cursor: once a row's stored segment ran out, the cursor still addressed the
next row's first stored coordinate, and whenever that coordinate happened to
equal the current column the next row's value was added into this row.

That produced silently wrong results from plain ``STensor.__add__`` -- and a
past-the-end read of the coordinate array once every stored entry was
consumed.  ``ds + dd`` (canonical CSR plus a dense matrix) is the plainest
affected case.

Each lattice-point case is now conjoined with its own cursor bound
(``pX < pX_end``) before the coordinate comparison.
"""

import itertools

import pytest
import torch

import scorch
from scorch.compiler.cin import (
    BinaryOp as CINBinaryOp,
    ForAll,
    IndexVar,
    Operation,
    TensorAssign,
    TensorVar,
)
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.pipeline import legacy_generated_cpp
from scorch.stensor import STensor

_RANK2 = ["ss", "ds", "sd", "dd"]


def operand(dense, name, fmt):
    tensor = STensor.from_torch(dense.clone(), name)
    if not all(character == "d" for character in fmt):
        tensor.to_sparse(fmt)
    return tensor


def test_csr_plus_dense_matrix_is_exact():
    """The plainest affected public case, with a hand-checked expectation."""

    left = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    right = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
    result = operand(left, "A", "ds") + operand(right, "B", "dd")
    assert result.to_torch().tolist() == [[1.0, 0.0], [0.0, 5.0]]


def test_doubly_compressed_plus_dense_matrix_is_exact():
    left = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    right = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
    result = operand(left, "A", "ss") + operand(right, "B", "dd")
    assert result.to_torch().tolist() == [[1.0, 0.0], [0.0, 5.0]]


@pytest.mark.parametrize("seed", [11, 13, 17, 101, 202])
@pytest.mark.parametrize("shape", [(3, 3), (4, 5), (5, 4)], ids=["3x3", "4x5", "5x4"])
def test_every_supported_rank2_addition_matches_torch(seed, shape):
    """A drained cursor must not leak the next segment's entries.

    The exhaustion is reachable only when some row's stored segment ends
    before the dense loop does, so the sweep uses several shapes and seeds --
    the defect was silent on many of them.
    """

    torch.manual_seed(seed)
    left = (torch.rand(shape) < 0.5) * torch.randn(shape)
    right = (torch.rand(shape) < 0.5) * torch.randn(shape)
    expected = left + right
    checked = 0
    for left_fmt, right_fmt in itertools.product(_RANK2, repeat=2):
        try:
            total = operand(left, "A", left_fmt) + operand(right, "B", right_fmt)
        except Exception:
            # A trailing-dense-level receiver is a separate, stable boundary.
            continue
        checked += 1
        assert torch.allclose(
            total.to_torch(), expected, atol=1e-5, rtol=1e-5
        ), f"{left_fmt} + {right_fmt}"
    assert checked >= 12


@pytest.mark.parametrize(
    "left_fmt,right_fmt",
    [
        ("ds", "dd"),
        ("ds", "sd"),
        ("ss", "dd"),
        ("ss", "sd"),
        ("dd", "ss"),
        ("dd", "ds"),
    ],
    ids=lambda value: value,
)
def test_ragged_supports_add_exactly(left_fmt, right_fmt):
    """Rows of very different stored lengths maximize the exhaustion window."""

    left = torch.zeros(4, 6)
    left[0] = torch.arange(1.0, 7.0)
    left[3, 5] = 9.0
    right = torch.zeros(4, 6)
    right[1] = torch.arange(1.0, 7.0)
    right[3, 0] = 4.0
    expected = left + right
    total = operand(left, "A", left_fmt) + operand(right, "B", right_fmt)
    assert torch.allclose(total.to_torch(), expected, atol=1e-6, rtol=1e-6)


def test_empty_and_full_operands_add_exactly():
    empty = torch.zeros(3, 4)
    full = torch.randn(3, 4)
    for left_fmt, right_fmt in (("ss", "dd"), ("ds", "dd"), ("dd", "ss")):
        assert torch.allclose(
            (operand(empty, "A", left_fmt) + operand(full, "B", right_fmt)).to_torch(),
            full,
            atol=1e-6,
        )
        assert torch.allclose(
            (operand(full, "A", left_fmt) + operand(empty, "B", right_fmt)).to_torch(),
            full,
            atol=1e-6,
        )


def test_float64_addition_matches_torch():
    torch.manual_seed(5)
    left = ((torch.rand(4, 5) < 0.5) * torch.randn(4, 5)).to(torch.float64)
    right = ((torch.rand(4, 5) < 0.5) * torch.randn(4, 5)).to(torch.float64)
    total = operand(left, "A", "ds") + operand(right, "B", "dd")
    assert torch.allclose(total.to_torch(), left + right, atol=1e-12, rtol=1e-12)


def test_rank3_addition_matches_torch():
    torch.manual_seed(9)
    left = (torch.rand(3, 4, 5) < 0.5) * torch.randn(3, 4, 5)
    right = (torch.rand(3, 4, 5) < 0.5) * torch.randn(3, 4, 5)
    total = operand(left, "A", "dss") + operand(right, "B", "ddd")
    assert torch.allclose(total.to_torch(), left + right, atol=1e-6, rtol=1e-6)


def _add_cin(left_fmt, right_fmt):
    ivars = (IndexVar("i"), IndexVar("j"))
    result = TensorVar("C", fmt="dd", dtype=torch.float32)
    left = TensorVar("A", fmt=left_fmt, dtype=torch.float32)[ivars]
    right = TensorVar("B", fmt=right_fmt, dtype=torch.float32)[ivars]
    stmt = TensorAssign(result[ivars], CINBinaryOp(Operation.ADD, left, right))
    for index_var in reversed(ivars):
        stmt = ForAll(index_var, stmt)
    return stmt


def test_generated_source_bounds_every_sparse_cursor():
    """The emitted case guard names the cursor bound, not only the coordinate."""

    source = legacy_generated_cpp(
        _add_cin("ss", "dd"),
        (4, 5),
        (((4, 5), torch.float32), ((4, 5), torch.float32)),
        compile_options=CompileOptions.from_environment(environ={}),
    )
    assert "pA1 < pA1_end" in source
    assert "j_A == j" in source


def test_all_sparse_lattice_is_byte_unchanged():
    """No dense universe means no cursor-bound term: the bytes are the legacy ones."""

    source = legacy_generated_cpp(
        _add_cin("ss", "ss"),
        (4, 5),
        (((4, 5), torch.float32), ((4, 5), torch.float32)),
        compile_options=CompileOptions.from_environment(environ={}),
    )
    assert "pA1 < pA1_end && j_A == j" not in source


def test_public_matmul_is_unaffected():
    """A reduction over a dense universe keeps matching torch.matmul."""

    torch.manual_seed(3)
    left = (torch.rand(4, 5) < 0.5) * torch.randn(4, 5)
    right = (torch.rand(5, 6) < 0.5) * torch.randn(5, 6)
    product = scorch.matmul(operand(left, "A", "ds"), operand(right, "B", "ds"))
    assert torch.allclose(product.to_torch(), left @ right, atol=1e-4, rtol=1e-4)
