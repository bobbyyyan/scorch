"""A shape signed int64 cannot hold is refused with the code ``shape_exceeds_int64``.

Scorch holds every extent and every element count as a signed 64-bit integer.  Each
boundary that holds a shape to that bound raises a ``ShapeCapacityError`` subclass of the
type it raised before the refusal had a code, so ``except`` clauses written for that type
keep working, and the message keeps the former text with the extent or count, its value
and the bound after it.  A zero extent holds no element: a shape with one has count 0,
whatever its other extents multiply to, and is not refused for its count.  Before this,
every one of these checks multiplied left to right and refused ``(N, N, 3, 0)`` at the
third extent.

The 32-bit position bound of the kernels is a different refusal with its own code
(``index_position_exceeds_int32``, ``tests/test_scorch/test_index_position_bound.py``);
the last part holds the two codes apart on the public routes of both compilers.
"""

from __future__ import annotations

import pickle

import pytest
import torch

import scorch
from scorch import STensor
from scorch.compiler import cin_analysis
from scorch.compiler.cin import BinaryOp, ForAll, IndexVar, Operation, TensorAssign
from scorch.compiler.cin import TensorVar
from scorch.compiler.diagnostics import (
    ShapeCapacityVerificationError,
    VerificationError,
)
from scorch.compiler.torch_cpp_abi import _validate_known_shape_range
from scorch.exceptions import (
    INT64_MAX,
    CompileSpecError,
    IndexCapacityError,
    ScorchError,
    ShapeCapacityError,
    ShapeCapacityKernelError,
    ShapeCapacityLayoutError,
    ShapeCapacitySpecError,
    TensorLayoutError,
    TensorValidationError,
)
from scorch.layout import TensorLayout, TensorSpec, _normalize_shape
from scorch.ops import _checked_runtime_shape

from tests.test_scorch.level_array_support import emitted_einsum_cpp

CODE = "shape_exceeds_int64"
BOUND = "the signed int64 bound 9223372036854775807 (2**63 - 1)"
#: The largest extent the kernels' 32-bit positions take (int32's maximum).
N = 2**31 - 1
#: Four extents of which the first three multiply past int64 and the last is zero.
ZERO_LAST = (N, N, 3, 0)
ZERO_PLACES = (ZERO_LAST, (N, N, 0, 3), (N, 0, N, 3), (0, N, N, 3))
#: Every extent within int64, the count 2**64.
COUNT_PAST = (2**62, 4)
EXTENT_PAST = (3, 2**63)


def assert_refusal(error, error_type, base, owner, mode, value):
    assert type(error) is error_type
    assert isinstance(error, ShapeCapacityError)
    assert isinstance(error, base)
    assert error.code == CODE
    assert error.owner == owner
    assert error.mode == mode
    assert error.value == value
    assert error.bound == INT64_MAX
    assert BOUND in str(error)
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is error_type
    assert str(restored) == str(error)
    assert (restored.code, restored.owner, restored.mode, restored.value) == (
        CODE,
        owner,
        mode,
        value,
    )


# --------------------------------------------------------------------------------------
# The layout, which every tensor construction and every compile-only spec goes through.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda shape: TensorLayout.from_physical_shape(shape, "d" * len(shape)),
        lambda shape: TensorLayout.from_logical_shape(shape, "d" * len(shape)),
        lambda shape: TensorLayout(
            shape, shape, "d" * len(shape), tuple(range(len(shape)))
        ),
    ],
    ids=("from_physical_shape", "from_logical_shape", "constructor"),
)
def test_a_layout_refuses_an_extent_and_a_count_past_int64_with_the_code(build):
    with pytest.raises(ShapeCapacityLayoutError) as caught:
        build(EXTENT_PAST)
    error = caught.value
    owner = error.owner
    assert owner in ("physical_shape", "logical_shape")
    assert_refusal(error, ShapeCapacityLayoutError, TensorLayoutError, owner, 1, 2**63)
    assert str(error) == (
        f"{owner}[1] exceeds signed int64: extent 1 is 9223372036854775808, "
        f"more than {BOUND}"
    )

    with pytest.raises(ShapeCapacityLayoutError) as caught:
        build(COUNT_PAST)
    error = caught.value
    owner = error.owner
    assert_refusal(
        error, ShapeCapacityLayoutError, TensorLayoutError, owner, None, 2**64
    )
    assert str(error) == (
        f"{owner} element count exceeds signed int64: shape (4611686018427387904, 4) "
        f"holds 18446744073709551616 elements, more than {BOUND}"
    )


@pytest.mark.parametrize("shape", ZERO_PLACES, ids=lambda s: "zero_at_%d" % s.index(0))
def test_a_layout_with_a_zero_extent_holds_no_element_whatever_the_others_are(shape):
    letters = "d" * len(shape)
    assert TensorLayout.from_physical_shape(shape, letters).physical_shape == shape
    assert TensorLayout.from_logical_shape(shape, letters).logical_shape == shape
    assert _normalize_shape(shape, "shape") == shape


def test_the_count_is_the_product_of_every_extent_exactly():
    # One below and one above the bound, with no extent past it.
    assert _normalize_shape((2**62 - 1, 2), "shape") == (2**62 - 1, 2)
    assert _normalize_shape((INT64_MAX, 1, 1), "shape") == (INT64_MAX, 1, 1)
    with pytest.raises(ShapeCapacityLayoutError) as caught:
        _normalize_shape((2**31, 2**31, 2), "shape")
    assert caught.value.value == 2**63


def test_an_extent_too_long_to_print_is_named_by_its_power_of_two():
    with pytest.raises(ShapeCapacityLayoutError) as caught:
        _normalize_shape((3, 1 << 20000), "shape")
    assert str(caught.value) == (
        f"shape[1] exceeds signed int64: extent 1 is 2**20000 or more, more than {BOUND}"
    )
    assert caught.value.value == 1 << 20000


def test_the_other_layout_refusals_keep_their_types_and_have_no_code():
    with pytest.raises(TensorLayoutError) as caught:
        _normalize_shape((3, -1), "shape")
    assert not isinstance(caught.value, ShapeCapacityError)
    with pytest.raises(scorch.TensorTypeError):
        _normalize_shape((3, 2.0), "shape")


# --------------------------------------------------------------------------------------
# The tensor constructions.
# --------------------------------------------------------------------------------------


def _empty_coo(shape):
    rank = len(shape)
    return dict(
        indices=torch.empty((rank, 0), dtype=torch.int64),
        values=torch.empty(0, dtype=torch.float64),
        shape=shape,
    )


def test_from_components_refuses_a_count_past_int64_and_builds_a_zero_extent():
    with pytest.raises(ShapeCapacityLayoutError) as caught:
        STensor.from_components(
            (N, N, 3, 1), "dddd", [[], [], [], []], torch.empty(0, dtype=torch.float64)
        )
    assert_refusal(
        caught.value,
        ShapeCapacityLayoutError,
        TensorLayoutError,
        "physical_shape",
        None,
        N * N * 3,
    )
    for shape in ZERO_PLACES:
        tensor = STensor.from_components(
            shape, "dddd", [[], [], [], []], torch.empty(0, dtype=torch.float64)
        )
        assert tuple(tensor.shape) == shape


def test_from_coo_refuses_a_count_past_int64_and_builds_a_zero_extent():
    with pytest.raises(ShapeCapacityLayoutError) as caught:
        STensor.from_coo(**_empty_coo((N, N, 3, 1)))
    assert caught.value.code == CODE
    assert caught.value.value == N * N * 3
    for shape in ZERO_PLACES:
        tensor = STensor.from_coo(**_empty_coo(shape))
        assert tuple(tensor.shape) == shape
        assert tensor.values.numel() == 0


def test_the_internal_constructor_refuses_with_the_code():
    with pytest.raises(ShapeCapacityLayoutError) as caught:
        STensor._from_parts(
            None,
            COUNT_PAST,
            "dd",
            [[], []],
            torch.empty(0, dtype=torch.float64),
        )
    assert caught.value.code == CODE


# --------------------------------------------------------------------------------------
# The compile-only spec and the runtime shape snapshot (shipped compiler).
# --------------------------------------------------------------------------------------


def test_a_compile_spec_keeps_the_code_of_its_layout_refusal():
    with pytest.raises(ShapeCapacitySpecError) as caught:
        TensorSpec("ooo", (2**22, 2**22, 2**22))
    assert_refusal(
        caught.value,
        ShapeCapacitySpecError,
        CompileSpecError,
        "logical_shape",
        None,
        2**66,
    )
    assert str(caught.value).startswith(
        "logical_shape element count exceeds signed int64: "
    )
    assert isinstance(caught.value.__cause__, ShapeCapacityLayoutError)
    for shape in ZERO_PLACES:
        assert TensorSpec("oooo", shape).shape == shape


def test_the_runtime_shape_snapshot_refuses_with_the_code():
    with pytest.raises(ShapeCapacitySpecError) as caught:
        _checked_runtime_shape(EXTENT_PAST, "result shape")
    assert_refusal(
        caught.value, ShapeCapacitySpecError, CompileSpecError, "result shape", 1, 2**63
    )
    assert str(caught.value).startswith(
        "result shape must contain nonnegative signed int64 extents: "
    )
    with pytest.raises(ShapeCapacitySpecError) as caught:
        _checked_runtime_shape(COUNT_PAST, "result shape")
    assert_refusal(
        caught.value,
        ShapeCapacitySpecError,
        CompileSpecError,
        "result shape",
        None,
        2**64,
    )
    for shape in ZERO_PLACES:
        assert _checked_runtime_shape(shape, "result shape") == shape
    with pytest.raises(CompileSpecError) as caught:
        _checked_runtime_shape((3, -1), "result shape")
    assert not isinstance(caught.value, ShapeCapacityError)


def test_the_kernel_abi_refuses_a_known_shape_with_the_code():
    with pytest.raises(ShapeCapacityKernelError) as caught:
        _validate_known_shape_range(EXTENT_PAST, "kernel result")
    assert_refusal(
        caught.value, ShapeCapacityKernelError, ValueError, "kernel result", 1, 2**63
    )
    assert not isinstance(caught.value, ScorchError)
    with pytest.raises(ShapeCapacityKernelError) as caught:
        _validate_known_shape_range(COUNT_PAST, "kernel tensor")
    assert str(caught.value).startswith(
        "known kernel tensor element count must fit signed int64: "
    )
    for shape in ZERO_PLACES:
        _validate_known_shape_range(shape, "kernel tensor")


def test_the_cin_shape_check_counts_a_zero_extent_as_no_element():
    for shape in ZERO_PLACES:
        assert cin_analysis._is_cin_shape(shape)
    assert not cin_analysis._is_cin_shape(COUNT_PAST)
    assert not cin_analysis._is_cin_shape(EXTENT_PAST)
    assert not cin_analysis._is_cin_shape((1,) * 65)
    assert not cin_analysis._is_cin_shape([3, 4])
    assert not cin_analysis._is_cin_shape((3, True))
    assert cin_analysis._is_cin_shape((INT64_MAX,))
    assert cin_analysis._is_cin_shape(())


# --------------------------------------------------------------------------------------
# The new compiler's boundaries.
# --------------------------------------------------------------------------------------


def _copy_program(letters):
    """``C[...] = A[...]``, every level of both ``letters``."""
    names = "ijkl"[: len(letters)]
    variables = [IndexVar(name) for name in names]
    a = TensorVar("A", fmt=letters)
    c = TensorVar("C", fmt=letters)
    statement = TensorAssign(c[tuple(variables)], a[tuple(variables)])
    for variable in reversed(variables):
        statement = ForAll(variable, statement)
    return statement


def _outer_program():
    """``D[i, j, k] = A[i] * B[j] * E[k]``, every tensor coordinate (COO)."""
    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a, b, e = (TensorVar(name, fmt="o") for name in "ABE")
    d = TensorVar("D", fmt="ooo")
    product = BinaryOp(Operation.MUL, BinaryOp(Operation.MUL, a[i], b[j]), e[k])
    return ForAll(i, ForAll(j, ForAll(k, TensorAssign(d[i, j, k], product))))


def test_the_new_compiler_refuses_a_result_past_int64_with_the_code():
    from scorch.compiler.loopir.pipeline import compile_cin_via_loopir

    side = 2**22
    bindings = (((side,), torch.float64),) * 3
    with pytest.raises(ShapeCapacityVerificationError) as caught:
        compile_cin_via_loopir(_outer_program(), (side, side, side), bindings)
    assert_refusal(
        caught.value,
        ShapeCapacityVerificationError,
        VerificationError,
        "runtime result shape",
        None,
        2**66,
    )
    assert str(caught.value).startswith(
        "runtime result shape element count exceeds signed int64: "
    )


def test_the_new_compiler_refuses_an_operand_past_int64_with_the_code():
    from scorch.compiler.loopir.pipeline import compile_cin_via_loopir

    shape = (2**22, 2**22, 2**22)
    with pytest.raises(ShapeCapacityVerificationError) as caught:
        compile_cin_via_loopir(_copy_program("ooo"), shape, ((shape, torch.float64),))
    assert_refusal(
        caught.value,
        ShapeCapacityVerificationError,
        VerificationError,
        "runtime input binding 0 shape",
        None,
        2**66,
    )
    with pytest.raises(ShapeCapacityVerificationError) as caught:
        compile_cin_via_loopir(
            _copy_program("oo"), (3, 4), (((3, 2**63), torch.float64),)
        )
    assert caught.value.mode == 1
    assert str(caught.value).startswith(
        "runtime input binding 0 shape must contain nonnegative int64 extents: "
    )


def test_the_request_identity_refuses_an_extent_past_int64_with_the_code():
    from scorch.compiler.loopir.plan_identity import _shape_payload

    with pytest.raises(ShapeCapacityVerificationError) as caught:
        _shape_payload([3, 1 << 20000], "loopir request result shape")
    assert caught.value.code == CODE
    assert caught.value.mode == 1
    assert "2**20000 or more" in str(caught.value)
    with pytest.raises(VerificationError) as caught:
        _shape_payload([3, -1], "loopir request result shape")
    assert not isinstance(caught.value, ShapeCapacityError)
    assert _shape_payload(list(ZERO_LAST), "loopir request result shape") == list(
        ZERO_LAST
    )


# --------------------------------------------------------------------------------------
# Both compilers on their public routes: the zero extent is emitted, a count past int64
# and an extent past the kernels' positions are refused with their two codes.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ZERO_PLACES, ids=lambda s: "zero_at_%d" % s.index(0))
def test_the_shipped_compiler_emits_a_zero_extent_operand_and_result(
    monkeypatch, shape
):
    extents = dict(zip("ijkl", shape))
    source = emitted_einsum_cpp(
        monkeypatch, "ijkl,ijkl->ijkl", ["oooo", "oooo"], extents, format="oooo"
    )
    assert "evaluate" in source


@pytest.mark.parametrize("shape", ZERO_PLACES, ids=lambda s: "zero_at_%d" % s.index(0))
def test_the_new_compiler_emits_a_zero_extent_operand_and_result(shape):
    from scorch.compiler.loopir.pipeline import compile_cin_via_loopir

    kernel = compile_cin_via_loopir(
        _copy_program("oooo"), shape, ((shape, torch.float64),)
    )
    assert "evaluate" in kernel.cpp_source


def test_the_shipped_compiler_refuses_a_result_past_int64(monkeypatch):
    """The shipped compiler checks a result's shape in the normalized CIN, where it has
    been coded ``invalid_cin_field`` at the result's ``TensorVar.shape``; that code stays.
    """

    side = 2**22
    with pytest.raises(VerificationError) as caught:
        emitted_einsum_cpp(
            monkeypatch,
            "i,j,k->ijk",
            ["o", "o", "o"],
            {"i": side, "j": side, "k": side},
            format="ooo",
        )
    codes = [diagnostic.code for diagnostic in caught.value.diagnostics]
    assert codes and set(codes) == {"invalid_cin_field"}
    assert "lhs/tensor/shape" in str(caught.value)


def _coo_vector(extent, dtype=torch.float64, index_dtype=torch.int64):
    return STensor.from_coo(
        indices=torch.tensor([[0, extent - 1]], dtype=index_dtype),
        values=torch.tensor([1.5, -2.0], dtype=dtype),
        shape=(extent,),
    )


def test_the_two_codes_are_apart_on_the_shipped_route():
    """``int64`` index arrays at the 32-bit position bound get the position code; a result
    whose every extent the kernels address but whose count passes int64 is refused before
    a kernel is built, and not with the position code."""

    with pytest.raises(IndexCapacityError) as caught:
        scorch.einsum("i,i->i", _coo_vector(N + 1), _coo_vector(N + 1), format="o")
    assert caught.value.code == "index_position_exceeds_int32"
    assert isinstance(caught.value, TensorValidationError)

    side = 2**22
    vectors = [_coo_vector(side) for _ in range(3)]
    with pytest.raises(VerificationError) as caught:
        scorch.einsum("i,j,k->ijk", *vectors, format="ooo")
    assert not isinstance(caught.value, IndexCapacityError)
    assert {d.code for d in caught.value.diagnostics} == {"invalid_cin_field"}


def test_the_two_codes_are_apart_on_the_new_compilers_route():
    from scorch.compiler.loopir.pipeline import execute_cin_via_loopir

    with pytest.raises(IndexCapacityError) as caught:
        execute_cin_via_loopir(_copy_program("o"), (N + 1,), _coo_vector(N + 1))
    assert caught.value.code == "index_position_exceeds_int32"

    # The executor snapshots the result shape as the shipped executor does
    # (ops._checked_runtime_shape), before it binds the program's metadata.
    side = 2**22
    vectors = [_coo_vector(side) for _ in range(3)]
    with pytest.raises(ShapeCapacitySpecError) as caught:
        execute_cin_via_loopir(_outer_program(), (side, side, side), *vectors)
    assert caught.value.code == CODE
    assert caught.value.owner == "result shape"
    assert not isinstance(caught.value, IndexCapacityError)
