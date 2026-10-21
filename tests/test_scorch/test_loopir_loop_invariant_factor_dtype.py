"""The hoisted loop-invariant factor takes the reduction's scalar type on the LoopIR route.

``C[i] = sum_k A[i, k] * B[i] * D[k]`` with a compressed result reduces over
``k`` into a scalar the LoopIR target declares with the result's dtype and
zeroes directly before the ``k`` loop.  ``B[i]`` does not change inside that
loop, so the loop-invariant factor pass moves it out: the loop accumulates
``A[i, k] * D[k]`` and the accumulator is multiplied by the factor once, after
the loop.

The temporary that holds the factor used to be declared ``float`` for every
dtype.  In a float64 kernel that rounded ``B[i]`` to single precision, a
relative error near 1e-8 in every result entry, which the suite's 1e-3
tolerance cannot see.  It is now declared with the accumulator's scalar type.

The first group of tests reads the generated C++ and compiles nothing.  The
second compiles kernels and compares float64 results with ``torch.einsum`` in
float64 at a relative error of 1e-12.  The third is one case whose answer is
exact, so that the lost bits are not a matter of tolerance: two stored ones
times ``1 + 2**-30`` is ``2 + 2**-29`` in float64 and exactly 2 through a
``float`` temporary.
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest
import torch

from scorch.compiler.cin import BinaryOp as CINBinaryOp
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.scheduler import Schedule
from scorch.stensor import STensor

ROWS, COLUMNS = 7, 9
FLOAT64_RELATIVE_TOLERANCE = 1e-12
FLOAT32_RELATIVE_TOLERANCE = 1e-5
# Each admits the program and reduces over ``k`` into a scalar accumulator.
MATRIX_FORMATS = ("ss", "ds", "sd")
_FACTOR_DECLARATION = re.compile(r"^\s*(\w+) (_inv_\d+) = B_val\[i\];$", re.MULTILINE)
_ACCUMULATOR_DECLARATION = re.compile(r"^\s*(\w+) C_reduction = 0\.0;$", re.MULTILINE)


def _options(regblock_enabled: bool) -> CompileOptions:
    return replace(
        CompileOptions.from_environment().with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(),
    )


def _row_scaled_reduction(matrix_format: str, dtype: torch.dtype) -> ForAll:
    """``C[i] += A[i, k] * B[i] * D[k]`` into a compressed vector."""

    i, k = IndexVar("i"), IndexVar("k")
    matrix = TensorVar("A", fmt=matrix_format, dtype=dtype)[i, k]
    row_scale = TensorVar("B", fmt="d", dtype=dtype)[i]
    column_weight = TensorVar("D", fmt="d", dtype=dtype)[k]
    product = CINBinaryOp(
        Operation.MUL, CINBinaryOp(Operation.MUL, matrix, row_scale), column_weight
    )
    result = TensorVar("C", fmt="s", dtype=dtype)[i]
    return ForAll(i, ForAll(k, TensorAssign(result, product, op=Operation.ADD)))


def _generated_cpp(
    matrix_format: str, dtype: torch.dtype, regblock_enabled: bool
) -> str:
    kernel = compile_cin_via_loopir(
        _row_scaled_reduction(matrix_format, dtype),
        (ROWS,),
        (((ROWS, COLUMNS), dtype), ((ROWS,), dtype), ((COLUMNS,), dtype)),
        compile_options=_options(regblock_enabled),
    )
    return kernel.cpp_source


# -- the factor's declaration (nothing compiled) -------------------------------


@pytest.mark.parametrize("regblock_enabled", (False, True), ids=("off", "on"))
@pytest.mark.parametrize("matrix_format", MATRIX_FORMATS)
@pytest.mark.parametrize(
    ("dtype", "scalar"),
    ((torch.float64, "double"), (torch.float32, "float")),
    ids=("float64", "float32"),
)
def test_hoisted_factor_is_declared_with_the_accumulator_scalar_type(
    dtype: torch.dtype, scalar: str, matrix_format: str, regblock_enabled: bool
) -> None:
    cpp = _generated_cpp(matrix_format, dtype, regblock_enabled)

    accumulators = _ACCUMULATOR_DECLARATION.findall(cpp)
    factors = _FACTOR_DECLARATION.findall(cpp)
    assert accumulators == [scalar]
    assert len(factors) == 1
    factor_type, factor_name = factors[0]
    assert factor_type == scalar
    assert f"C_reduction *= {factor_name};" in cpp
    # The factor left the loop: the accumulation no longer reads it.
    assert not re.search(r"C_reduction \+= .*B_val", cpp)


@pytest.mark.parametrize("regblock_enabled", (False, True), ids=("off", "on"))
@pytest.mark.parametrize("matrix_format", MATRIX_FORMATS)
def test_float64_kernel_is_the_float32_kernel_with_the_scalar_respelled(
    matrix_format: str, regblock_enabled: bool
) -> None:
    float32_cpp = _generated_cpp(matrix_format, torch.float32, regblock_enabled)
    float64_cpp = _generated_cpp(matrix_format, torch.float64, regblock_enabled)

    assert "double" not in float32_cpp
    assert float64_cpp == (
        float32_cpp.replace("float", "double").replace("kFloat32", "kFloat64")
    )


# -- float64 execution against a float64 reference (compiles kernels) ----------


def _operands(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(29)
    matrix = (
        torch.rand((ROWS, COLUMNS), generator=generator, dtype=torch.float64) + 0.5
    ) * (torch.rand((ROWS, COLUMNS), generator=generator) < 0.6)
    assert bool((matrix != 0).any(dim=1).all())
    row_scale = torch.rand((ROWS,), generator=generator, dtype=torch.float64) + 0.5
    column_weight = (
        torch.rand((COLUMNS,), generator=generator, dtype=torch.float64) + 0.5
    )
    return matrix.to(dtype), row_scale.to(dtype), column_weight.to(dtype)


def _max_relative_error(
    matrix_format: str, dtype: torch.dtype, regblock_enabled: bool
) -> float:
    matrix, row_scale, column_weight = _operands(dtype)
    expected = torch.einsum(
        "ik,i,k->i", matrix.double(), row_scale.double(), column_weight.double()
    )
    result, kernel = execute_cin_via_loopir(
        _row_scaled_reduction(matrix_format, dtype),
        (ROWS,),
        STensor.from_torch(matrix.clone(), "A").to_sparse(matrix_format),
        STensor.from_torch(row_scale.clone(), "B").to_dense(),
        STensor.from_torch(column_weight.clone(), "D").to_dense(),
        compile_options=_options(regblock_enabled),
    )
    # The kernel under test is one in which the pass fired.
    assert len(_FACTOR_DECLARATION.findall(kernel.cpp_source)) == 1
    actual = result.to_torch(in_place=False).double()
    assert actual.shape == expected.shape
    return float(((actual - expected).abs() / expected.abs()).max())


@pytest.mark.parametrize("regblock_enabled", (False, True), ids=("off", "on"))
@pytest.mark.parametrize("matrix_format", ("ss", "ds"))
def test_float64_row_scaled_reduction_matches_float64_einsum(
    matrix_format: str, regblock_enabled: bool
) -> None:
    error = _max_relative_error(matrix_format, torch.float64, regblock_enabled)

    assert error <= FLOAT64_RELATIVE_TOLERANCE


@pytest.mark.parametrize("matrix_format", ("ss", "ds"))
def test_float32_row_scaled_reduction_matches_einsum_at_float32_precision(
    matrix_format: str,
) -> None:
    error = _max_relative_error(matrix_format, torch.float32, False)

    assert error <= FLOAT32_RELATIVE_TOLERANCE


# -- an exact case: the factor keeps its low bits (compiles kernels) -----------
#
# From the independent review of a20fe61 (/Users/bobby/loopir-review, finding
# 2).  The sum has two terms and one factor, every operation is exact in
# float64, and the factor differs from 1 only below single precision.


def _row_scaled_sum(dtype: torch.dtype) -> ForAll:
    """``C[i] += A[i, k] * B[i]``, dense rows and compressed columns, into a compressed vector."""

    i, k = IndexVar("i"), IndexVar("k")
    matrix = TensorVar("A", fmt="ds", dtype=dtype)[i, k]
    row_scale = TensorVar("B", fmt="d", dtype=dtype)[i]
    result = TensorVar("C", fmt="s", dtype=dtype)[i]
    product = CINBinaryOp(Operation.MUL, matrix, row_scale)
    return ForAll(i, ForAll(k, TensorAssign(result, product, op=Operation.ADD)))


@pytest.mark.parametrize("regblock_enabled", (False, True), ids=("off", "on"))
@pytest.mark.parametrize(
    "index_dtype", (torch.int32, torch.int64), ids=("int32", "int64")
)
def test_float64_factor_keeps_the_bits_single_precision_drops(
    index_dtype: torch.dtype, regblock_enabled: bool
) -> None:
    factor = 1.0 + 2.0**-30
    assert float(torch.tensor(factor, dtype=torch.float32)) == 1.0
    matrix = STensor.from_components(
        (1, 2),
        "ds",
        [
            [],
            [
                torch.tensor([0, 2], dtype=index_dtype),
                torch.tensor([0, 1], dtype=index_dtype),
            ],
        ],
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        name="A",
    )
    row_scale = STensor.from_components(
        (1,), "d", [[]], torch.tensor([factor], dtype=torch.float64), name="B"
    )

    result, kernel = execute_cin_via_loopir(
        _row_scaled_sum(torch.float64),
        (1,),
        matrix,
        row_scale,
        compile_options=_options(regblock_enabled),
    )

    # The kernel under test is one in which the pass fired.
    assert [scalar for scalar, _ in _FACTOR_DECLARATION.findall(kernel.cpp_source)] == [
        "double"
    ]
    position, coordinate = result.storage.index.mode_indices[0]
    assert position.tolist() == [0, 1]
    assert coordinate.tolist() == [0]
    assert result.storage.value.tolist() == [2.0 + 2.0**-29]
