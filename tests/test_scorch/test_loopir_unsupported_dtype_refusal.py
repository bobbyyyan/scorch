"""The new compiler refuses an unsupported element type with its structured code.

``compile_cin_via_loopir`` serializes the request before it lowers it, and that
serialization encodes float32 and float64 only.  An ``int64`` operand reached it
first and came back as ``VerificationError("loopir request input dtypes must be
exact supported torch.dtype values")`` with no diagnostics -- a malformed-request
error, with no code, for a well-formed program the lowering would have refused as
``unsupported_dtype``.  The compile entry now asks the lowering's own
classification before it serializes.

The supported set has since grown from float32 and float64 to the shipped
compiler's six element types (int64, int32, int8 and uint8 as well); the types
outside it are refused as before.
"""

from dataclasses import replace

import pytest
import torch

from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import VerificationError
from scorch.compiler.loopir.lower_cin import LoopIRLoweringError
from scorch.compiler.loopir.nodes import ScalarType
from scorch.compiler.loopir.pipeline import compile_cin_via_loopir
from scorch.compiler.scheduler import Schedule
from scorch.exceptions import UnsupportedDtypePromotionError

UNSUPPORTED = (
    torch.int16,
    torch.bool,
    torch.float16,
    torch.complex64,
)
SUPPORTED = (
    (torch.float32, ScalarType.FLOAT32, "float"),
    (torch.float64, ScalarType.FLOAT64, "double"),
    (torch.int64, ScalarType.INT64, "int64_t"),
    (torch.int32, ScalarType.INT32, "int32_t"),
    (torch.int8, ScalarType.INT8, "int8_t"),
    (torch.uint8, ScalarType.UINT8, "uint8_t"),
)


def row_sum(dtype):
    """``y[a] = sum_b A[a, b]`` over a compressed ``A``, with a dense result."""

    a, b = IndexVar("a"), IndexVar("b")
    operand = TensorVar("A", fmt="ss", dtype=dtype)
    result = TensorVar("y", fmt="d", dtype=dtype)
    assign = TensorAssign(result[a], operand[a, b], op=Operation.ADD)
    return ForAll(a, ForAll(b, assign))


def spmv(matrix_dtype, vector_dtype):
    """``y[a] = sum_b A[a, b] * x[b]``."""

    a, b = IndexVar("a"), IndexVar("b")
    matrix = TensorVar("A", fmt="ds", dtype=matrix_dtype)
    vector = TensorVar("x", fmt="d", dtype=vector_dtype)
    result = TensorVar("y", fmt="d", dtype=matrix_dtype)
    assign = TensorAssign(result[a], matrix[a, b] * vector[b], op=Operation.ADD)
    return ForAll(a, ForAll(b, assign))


def options(register_blocking, scheduled):
    chosen = CompileOptions.from_environment().with_regblock_enabled(register_blocking)
    return replace(chosen, requested_schedule=Schedule()) if scheduled else chosen


@pytest.mark.parametrize("scheduled", (False, True), ids=("unscheduled", "scheduled"))
@pytest.mark.parametrize("register_blocking", (False, True), ids=("off", "on"))
@pytest.mark.parametrize("dtype", UNSUPPORTED, ids=str)
def test_unsupported_operand_dtype_is_refused_with_its_code(
    dtype, register_blocking, scheduled
):
    with pytest.raises(LoopIRLoweringError) as refusal:
        compile_cin_via_loopir(
            row_sum(dtype),
            (3,),
            (((3, 4), dtype),),
            compile_options=options(register_blocking, scheduled),
        )

    assert refusal.value.defect.code == "unsupported_dtype"
    assert f"dtype {dtype}" in refusal.value.defect.message


@pytest.mark.parametrize(
    ("matrix_dtype", "vector_dtype", "named"),
    [
        (torch.float64, torch.int16, "'x'"),
        (torch.int16, torch.float32, "'A'"),
    ],
)
def test_one_unsupported_operand_among_supported_ones_is_refused(
    matrix_dtype, vector_dtype, named
):
    with pytest.raises(LoopIRLoweringError) as refusal:
        compile_cin_via_loopir(
            spmv(matrix_dtype, vector_dtype),
            (3,),
            (((3, 4), matrix_dtype), ((4,), vector_dtype)),
            compile_options=options(False, True),
        )

    assert refusal.value.defect.code == "unsupported_dtype"
    assert f"tensor {named}" in refusal.value.defect.message


@pytest.mark.parametrize(
    ("dtype", "spelled"),
    [(dtype, spelled) for dtype, _, spelled in SUPPORTED],
    ids=[str(dtype) for dtype, _, _ in SUPPORTED],
)
def test_supported_dtypes_lower(dtype, spelled):
    kernel = compile_cin_via_loopir(
        row_sum(dtype),
        (3,),
        (((3, 4), dtype),),
        compile_options=options(False, True),
    )

    assert f"{spelled}*" in kernel.cpp_source


@pytest.mark.parametrize(
    ("matrix_dtype", "vector_dtype", "promoted"),
    [
        (torch.float64, torch.int64, ScalarType.FLOAT64),
        (torch.int32, torch.float32, ScalarType.FLOAT32),
    ],
)
def test_two_supported_dtypes_in_one_program_compute_in_their_promoted_type(
    matrix_dtype, vector_dtype, promoted
):
    # Refused with ``mixed_dtype`` until 2026-10-12.  The result is bound to
    # the operands' promoted type, as the shipped entry binds it, and each
    # read keeps its operand's type; test_loopir_mixed_dtype_sums.py runs sums
    # of such pairs on both compilers.  The public matmul and einsum raise for
    # a mixed product before either compiler sees it (ken's 0036).
    kernel = compile_cin_via_loopir(
        spmv(matrix_dtype, vector_dtype),
        (3,),
        (((3, 4), matrix_dtype), ((4,), vector_dtype)),
        compile_options=options(False, True),
    )
    declared = {decl.name: decl.dtype for decl in kernel.lowering.program.tensors}
    assert declared == {
        "A": dict((t, s) for t, s, _ in SUPPORTED)[matrix_dtype],
        "x": dict((t, s) for t, s, _ in SUPPORTED)[vector_dtype],
        "y": promoted,
    }


def test_int8_with_uint8_is_refused_before_lowering():
    """They promote to int16, which Scorch does not store."""

    with pytest.raises(UnsupportedDtypePromotionError, match="needs torch.int16"):
        compile_cin_via_loopir(
            spmv(torch.int8, torch.uint8),
            (3,),
            (((3, 4), torch.int8), ((4,), torch.uint8)),
            compile_options=options(False, True),
        )


def test_a_value_that_is_not_a_dtype_is_still_a_malformed_request():
    with pytest.raises(VerificationError, match="must be an exact torch.dtype"):
        compile_cin_via_loopir(
            row_sum(torch.float64),
            (3,),
            (((3, 4), "torch.int64"),),
            compile_options=options(False, True),
        )


def test_the_classification_has_one_owner():
    from scorch.compiler.loopir.lower_cin import supported_scalar_type

    for dtype, scalar_type, _ in SUPPORTED:
        assert (
            supported_scalar_type(TensorVar("A", fmt="ss", dtype=dtype)) is scalar_type
        )
    for dtype in UNSUPPORTED:
        with pytest.raises(LoopIRLoweringError) as refusal:
            supported_scalar_type(TensorVar("A", fmt="ss", dtype=dtype))
        assert refusal.value.defect.code == "unsupported_dtype"
