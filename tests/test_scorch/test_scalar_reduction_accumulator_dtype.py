"""The scalar reduction accumulator of an all-coordinate result takes its element type.

When every level of the result is a coordinate level and the innermost loop is
a reduction, ``IterationLattice`` accumulates the reduction into a scalar local
named ``_accum`` and appends it once per result entry.  That local used to be
declared ``float`` for every dtype.  A float64 kernel truncated each partial sum
to single precision, a relative error near 1e-7 that the suite's 1e-3 tolerance
cannot see, and an integer kernel was exact only while every partial sum stayed
below 2**24.  The accumulator is now declared with the result's element type:
``double _accum = 0.0;``, ``int64_t _accum = 0;`` and so on.  float32 keeps
``float _accum = 0.0f;`` byte for byte.

The first group of tests reads the lowered LLIR and the generated C++ and
compiles nothing.  The second group needs the loop-invariant factor pass to
declare its hoisted factor with the accumulator's scalar type as well: with a
``double`` accumulator and a ``float`` factor the product is still rounded to
single precision.  The third group compiles kernels and compares float64
results with ``torch.einsum`` in float64 at a relative error of 1e-12, and
integer results with ``torch.einsum`` on int64 for exact equality, at sums
above 2**24.
"""

from __future__ import annotations

from typing import List, cast

import pytest
import torch

import scorch
from scorch import STensor
from scorch.compiler import llir
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.diagnostics import UnsupportedFeature
from scorch.compiler.scheduler import Scheduler, regblock_force

FLOAT64_RELATIVE_TOLERANCE = 1e-12
FLOAT32_RELATIVE_TOLERANCE = 1e-5


def _all_coordinate_sddmm(dtype: torch.dtype) -> ForAll:
    """``Sampled[r, c] += Mask[r, c] * Query[r, q] * Key[c, q]``, all-COO result."""

    row, column, reduction = IndexVar("r"), IndexVar("c"), IndexVar("q")
    result = TensorVar("Sampled", fmt="oo", dtype=dtype)
    mask = TensorVar("Mask", fmt="oo", dtype=dtype)
    query = TensorVar("Query", fmt="dd", dtype=dtype)
    key = TensorVar("Key", fmt="dd", dtype=dtype)
    assignment = TensorAssign(
        result[row, column],
        mask[row, column] * query[row, reduction] * key[column, reduction],
        op=Operation.ADD,
    )
    return cast(
        ForAll,
        Scheduler.auto_schedule(
            ForAll(row, ForAll(column, ForAll(reduction, assignment)))
        ),
    )


def _all_coordinate_row_sum(dtype: torch.dtype) -> ForAll:
    """``Sums[r] += Matrix[r, c]``: a reduction over a level of the driving operand."""

    row, column = IndexVar("r"), IndexVar("c")
    result = TensorVar("Sums", fmt="o", dtype=dtype)
    matrix = TensorVar("Matrix", fmt="oo", dtype=dtype)
    assignment = TensorAssign(result[row], matrix[row, column], op=Operation.ADD)
    # Left in its declared loop order: the automatic scheduler would route this
    # reduction through a workspace instead of the scalar accumulator.
    return ForAll(row, ForAll(column, assignment))


def _var_inits_named(value: object, name: str) -> List[llir.VarInit]:
    found: List[llir.VarInit] = []
    if type(value) is llir.VarInit and cast(llir.VarInit, value).var.name == name:
        found.append(cast(llir.VarInit, value))
    if isinstance(value, llir.Node):
        for child in vars(value).values():
            found.extend(_var_inits_named(child, name))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.extend(_var_inits_named(child, name))
    return found


def _lower(statement: ForAll) -> tuple[object, str]:
    with regblock_force(False):
        lowered = CINLowerer().lower_IndexStmt(statement)
        return lowered, LLIRLowerer().lower_llir(lowered)


# -- the accumulator's declaration (nothing compiled) --------------------------


@pytest.mark.parametrize("build", (_all_coordinate_sddmm,))
def test_float64_result_declares_a_double_accumulator(build) -> None:
    lowered, cpp = _lower(build(torch.float64))

    declarations = _var_inits_named(lowered, "_accum")
    assert len(declarations) == 1
    assert declarations[0].var == llir.Var("_accum", llir.DataType.FLOAT64)
    assert type(declarations[0].value) is llir.Literal
    assert cast(llir.Literal, declarations[0].value).value == "0.0"
    assert cast(llir.Literal, declarations[0].value).data_type is llir.DataType.FLOAT64
    assert "double _accum = 0.0;" in cpp
    assert "float _accum" not in cpp


_ELEMENT_TYPES = (
    (torch.float32, llir.DataType.FLOAT32, "float", "0.0f"),
    (torch.int64, llir.DataType.INT64, "int64_t", "0"),
    (torch.int32, llir.DataType.INT32, "int32_t", "0"),
    (torch.int8, llir.DataType.INT8, "int8_t", "0"),
    (torch.uint8, llir.DataType.UINT8, "uint8_t", "0"),
)


@pytest.mark.parametrize("build", (_all_coordinate_sddmm,))
@pytest.mark.parametrize(
    ("dtype", "scalar_type", "spelling", "zero"),
    _ELEMENT_TYPES,
    ids=[str(dtype) for dtype, _, _, _ in _ELEMENT_TYPES],
)
def test_every_dtype_declares_the_accumulator_with_its_element_type(
    build, dtype, scalar_type, spelling, zero
) -> None:
    lowered, cpp = _lower(build(dtype))

    declarations = _var_inits_named(lowered, "_accum")
    assert len(declarations) == 1
    assert declarations[0].var == llir.Var("_accum", scalar_type)
    assert cast(llir.Literal, declarations[0].value).value == zero
    assert cast(llir.Literal, declarations[0].value).data_type is scalar_type
    assert f"{spelling} _accum = {zero};" in cpp
    if dtype is not torch.float32:
        assert "float _accum" not in cpp


@pytest.mark.parametrize(
    ("dtype", "spelling"),
    ((torch.int64, "int64_t"), (torch.int32, "int32_t")),
    ids=("int64", "int32"),
)
def test_integer_all_coordinate_sddmm_hoists_a_factor_of_its_element_type(
    dtype, spelling
) -> None:
    _, cpp = _lower(_all_coordinate_sddmm(dtype))

    assert f"    {spelling} _accum = 0;\n" in cpp
    assert f"    {spelling} _inv_17 = Mask_val[pMask0];\n" in cpp
    assert "      _accum += _Query_val_ptr[q] * _Key_val_ptr[q];\n" in cpp
    assert "    _accum *= _inv_17;" in cpp
    assert "float" not in cpp[cpp.index("evaluate(") :]


@pytest.mark.parametrize(
    "dtype",
    (torch.float64, torch.float32, torch.int32, torch.int64, torch.int8, torch.uint8),
    ids=str,
)
def test_all_coordinate_row_sum_is_refused(dtype) -> None:
    # The reduced index ``c`` is a level of ``Matrix``, whose stored entries one
    # flat loop visits.  That loop appended an entry of ``Sums`` per stored entry of
    # ``Matrix`` -- a row repeated once per entry, each holding one term -- from a
    # parallel loop that advanced the shared position ``pSums0``.  The program was
    # never executed here, only read for its accumulator's declaration; it is
    # refused, so the all-coordinate SDDMM alone declares the accumulator.
    with pytest.raises(UnsupportedFeature) as refusal:
        _lower(_all_coordinate_row_sum(dtype))

    assert [diagnostic.code for diagnostic in refusal.value.diagnostics] == [
        "unsupported_coordinate_result_reduction"
    ]


# -- together with the loop-invariant factor pass (nothing compiled) -----------


def test_float64_all_coordinate_sddmm_hoists_a_double_factor() -> None:
    lowered, cpp = _lower(_all_coordinate_sddmm(torch.float64))

    factor_declarations = _var_inits_named(lowered, "_inv_17")
    assert len(factor_declarations) == 1
    assert factor_declarations[0].var == llir.Var("_inv_17", llir.DataType.FLOAT64)
    assert "float _inv_" not in cpp
    assert (
        "    double _inv_17 = Mask_val[pMask0];\n"
        "    #pragma omp simd\n"
        "    for (int64_t q = 0; q < Query1_size; q++) {\n"
        "      // Resolve dense coordinates\n"
        "      _accum += _Query_val_ptr[q] * _Key_val_ptr[q];\n"
        "    }\n"
        "    _accum *= _inv_17;"
    ) in cpp


def test_float32_all_coordinate_sddmm_still_hoists_a_float_factor() -> None:
    float32_lowered, float32_cpp = _lower(_all_coordinate_sddmm(torch.float32))
    _, float64_cpp = _lower(_all_coordinate_sddmm(torch.float64))

    factor_declarations = _var_inits_named(float32_lowered, "_inv_17")
    assert len(factor_declarations) == 1
    assert factor_declarations[0].var == llir.Var("_inv_17", llir.DataType.FLOAT32)
    assert "    float _inv_17 = Mask_val[pMask0];\n" in float32_cpp
    assert "double" not in float32_cpp
    # The two kernels are the same text with the scalar type respelled.
    assert float64_cpp == (
        float32_cpp.replace("0.0f", "0.0")
        .replace("float", "double")
        .replace("kFloat32", "kFloat64")
    )


# -- float64 execution against a float64 reference (compiles kernels) ----------


def _max_relative_error(result: object, expected: torch.Tensor) -> float:
    actual = result.to_torch(in_place=False) if isinstance(result, STensor) else result
    actual = cast(torch.Tensor, actual).double()
    assert actual.shape == expected.shape
    stored = expected != 0
    assert bool(stored.any())
    assert bool((actual[~stored] == 0).all())
    return float(((actual - expected).abs() / expected.abs())[stored].max())


def _sddmm_operands(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(3)
    rows, columns, rank = 7, 6, 5
    mask = (
        torch.rand((rows, columns), generator=generator, dtype=torch.float64) + 0.5
    ) * (torch.rand((rows, columns), generator=generator) < 0.5)
    query = torch.rand((rows, rank), generator=generator, dtype=torch.float64) + 0.5
    key = torch.rand((columns, rank), generator=generator, dtype=torch.float64) + 0.5
    return mask.to(dtype), query.to(dtype), key.to(dtype)


@pytest.mark.parametrize("mask_format", ("oo", "ds"))
def test_float64_coordinate_result_sddmm_matches_float64_einsum(mask_format) -> None:
    mask, query, key = _sddmm_operands(torch.float64)
    expected = torch.einsum("ij,ik,jk->ij", mask, query, key)

    result = scorch.einsum(
        "ij,ik,jk->ij",
        STensor.from_torch(mask.clone(), "Mask").to_sparse(mask_format),
        STensor.from_torch(query.clone(), "Query").to_dense(),
        STensor.from_torch(key.clone(), "Key").to_dense(),
        format="oo",
        use_cache=False,
    )

    assert _max_relative_error(result, expected) <= FLOAT64_RELATIVE_TOLERANCE


@pytest.mark.parametrize("mask_format", ("oo", "ds"))
def test_float64_sddmm_with_a_mask_of_ones_matches_float64_einsum(mask_format) -> None:
    # The hoisted factor is the mask's stored value.  A one is exact in single
    # precision, so the factor loses nothing whatever type it is declared
    # with, and this result depends on the accumulator's declaration alone.
    mask, query, key = _sddmm_operands(torch.float64)
    mask = (mask != 0).to(torch.float64)
    expected = torch.einsum("ij,ik,jk->ij", mask, query, key)

    result = scorch.einsum(
        "ij,ik,jk->ij",
        STensor.from_torch(mask.clone(), "Mask").to_sparse(mask_format),
        STensor.from_torch(query.clone(), "Query").to_dense(),
        STensor.from_torch(key.clone(), "Key").to_dense(),
        format="oo",
        use_cache=False,
    )

    assert _max_relative_error(result, expected) <= FLOAT64_RELATIVE_TOLERANCE


@pytest.mark.parametrize("mask_format", ("oo", "ds"))
def test_float32_coordinate_result_sddmm_matches_einsum_at_float32_precision(
    mask_format,
) -> None:
    mask, query, key = _sddmm_operands(torch.float32)
    expected = torch.einsum("ij,ik,jk->ij", mask.double(), query.double(), key.double())

    result = scorch.einsum(
        "ij,ik,jk->ij",
        STensor.from_torch(mask.clone(), "Mask").to_sparse(mask_format),
        STensor.from_torch(query.clone(), "Query").to_dense(),
        STensor.from_torch(key.clone(), "Key").to_dense(),
        format="oo",
        use_cache=False,
    )

    assert _max_relative_error(result, expected) <= FLOAT32_RELATIVE_TOLERANCE


# -- integer results, exact, at sums above 2**24 (compiles kernels) ------------
#
# float holds every integer up to 2**24 exactly and loses the low bits above
# it, so these sums are chosen above 2**24.  The int64 sums are also above
# 2**31, so an int32 accumulator would fail them too.


def _integer_sddmm_operands(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(13)
    rows, columns, rank = 5, 4, 5
    largest = 100_000 if dtype is torch.int64 else 6_000
    mask = torch.randint(1, 4, (rows, columns), generator=generator) * (
        torch.rand((rows, columns), generator=generator) < 0.7
    )
    query = torch.randint(largest // 2, largest, (rows, rank), generator=generator)
    key = torch.randint(largest // 2, largest, (columns, rank), generator=generator)
    return mask.to(dtype), query.to(dtype), key.to(dtype)


@pytest.mark.parametrize("mask_format", ("oo", "ds"))
@pytest.mark.parametrize("dtype", (torch.int64, torch.int32), ids=("int64", "int32"))
def test_integer_coordinate_result_sddmm_equals_int64_einsum(
    dtype, mask_format
) -> None:
    mask, query, key = _integer_sddmm_operands(dtype)
    expected = torch.einsum(
        "ij,ik,jk->ij", mask.to(torch.int64), query.to(torch.int64), key.to(torch.int64)
    )
    stored = expected != 0
    assert int(expected[stored].min()) > 2**24
    if dtype is torch.int64:
        assert int(expected[stored].min()) > 2**31
    else:
        assert int(expected.max()) < 2**31

    result = scorch.einsum(
        "ij,ik,jk->ij",
        STensor.from_torch(mask.clone(), "Mask").to_sparse(mask_format),
        STensor.from_torch(query.clone(), "Query").to_dense(),
        STensor.from_torch(key.clone(), "Key").to_dense(),
        format="oo",
        use_cache=False,
    )

    actual = result.to_torch(in_place=False)
    assert actual.dtype is dtype
    assert torch.equal(actual.to(torch.int64), expected)


@pytest.mark.parametrize("matrix_format", ("oo",))
@pytest.mark.parametrize("dtype", (torch.int64, torch.int32), ids=("int64", "int32"))
def test_integer_row_sum_through_einsum_equals_int64_sum(dtype, matrix_format) -> None:
    # A control that must pass before and after: the public row sum reduces
    # through a workspace typed with the result's element type, not through
    # the scalar accumulator, so its result was already exact.
    #
    # Only a COO operand is used.  For a matrix stored ``ss`` einsum first
    # re-lays the operand out into its loop order, through a kernel whose
    # workspace is declared ``float`` whatever the dtype.  That rounds every
    # int64 value above 2**24 before the reduction sees it: a separate defect
    # in the re-layout, which this control would then measure instead.
    generator = torch.Generator().manual_seed(17)
    rows, columns = 6, 40
    largest = 10**10 if dtype is torch.int64 else 4_000_000
    matrix = torch.randint(largest // 2, largest, (rows, columns), generator=generator)
    matrix = matrix * (torch.rand((rows, columns), generator=generator) < 0.6)
    expected = matrix.sum(dim=1)
    assert int(expected.min()) > 2**24

    result = scorch.einsum(
        "ij->i",
        STensor.from_torch(matrix.to(dtype).clone(), "Matrix").to_sparse(matrix_format),
        format=matrix_format[0],
        use_cache=False,
    )

    actual = result.to_torch(in_place=False)
    assert actual.dtype is dtype
    assert torch.equal(actual.to(torch.int64), expected)
