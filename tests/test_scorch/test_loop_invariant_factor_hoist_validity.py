"""Loop-invariant factor hoisting in generated kernels: kept where valid, absent where not.

The pass rewrites ``acc += s * x`` inside a loop as ``acc += x`` followed by
``acc *= s`` after the loop.  That is the same sum only for an accumulator that
is zero on entry to the loop.

Valid: the all-coordinate SDDMM kernel zeroes ``_accum`` directly before the
reduction loop, once per result entry.  Its hoisted loop must stay exactly as
it is; the first group of tests pins the emitted text.  So does a LoopIR
reduction whose result keeps both outer indices: its accumulator is zeroed
inside the middle loop, and the second group pins that it still hoists.

Not valid: ``C[i] = sum_jk A[i, j, k] * B[j] * D[k]`` into a compressed vector on
the LoopIR route zeroes its accumulator once per ``i`` and reduces over ``j``
and ``k``.  ``B[j]`` is invariant in the ``k`` loop, but the accumulator already
holds the sums of the earlier ``j`` when that loop starts, so multiplying it by
``B[j]`` afterwards rescales them too.  The pass used to do exactly that, and
the kernel returned results wrong by tens of percent in float32 and float64.
The third group pins that the factor now stays inside the loop, and the fourth
executes the kernel against ``torch.einsum``.

Not valid either, and smaller: ``C[i] = sum_kr A[i, k, r] * B[i, k]`` with dense
rows of ``A`` into a compressed vector.  ``B[i, k]`` is invariant in the ``r``
loop and the accumulator is zeroed outside the ``k`` loop.  With four stored
ones and ``B = [[2, 3]]`` the answer is 10; the hoisted form computes
``(0 + 2) * 2 = 4`` and then ``(4 + 2) * 3 = 18``.  An independent review of
a20fe61 found this program returning 18.  The fifth group pins its emitted loop
and its exact result.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import cast

import pytest
import torch

from scorch.compiler.cin import BinaryOp as CINBinaryOp
from scorch.compiler.cin import ForAll, IndexVar, Operation, TensorAssign, TensorVar
from scorch.compiler.cin_lowerer import CINLowerer
from scorch.compiler.codegen import LLIRLowerer
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.loopir.pipeline import (
    compile_cin_via_loopir,
    execute_cin_via_loopir,
)
from scorch.compiler.scheduler import Schedule, Scheduler, regblock_force
from scorch.stensor import STensor
from tests.test_scorch.openmp_shared_writes import shared_scalar_writes

FLOAT64_RELATIVE_TOLERANCE = 1e-12
FLOAT32_RELATIVE_TOLERANCE = 2e-5
EXTENTS = (6, 5, 4)


# -- valid: the SDDMM kernel's hoisted loop keeps its bytes (nothing compiled) --


def _all_coordinate_result_sddmm(mask_format: str) -> ForAll:
    row, column, reduction = IndexVar("r"), IndexVar("c"), IndexVar("q")
    result = TensorVar("Sampled", fmt="oo", dtype=torch.float32)
    mask = TensorVar("Mask", fmt=mask_format, dtype=torch.float32)
    query = TensorVar("Query", fmt="dd", dtype=torch.float32)
    key = TensorVar("Key", fmt="dd", dtype=torch.float32)
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


_COORDINATE_MASK_HOISTED_LOOP = (
    "    float _accum = 0.0f;\n"
    "     \n"
    "    const float* __restrict__ _Key_val_ptr = &Key_val[pKey0 * Key1_size];\n"
    "    const float* __restrict__ _Query_val_ptr = "
    "&Query_val[pQuery0 * Query1_size];\n"
    "    float _inv_17 = Mask_val[pMask0];\n"
    "    #pragma omp simd\n"
    "    for (int64_t q = 0; q < Query1_size; q++) {\n"
    "      // Resolve dense coordinates\n"
    "      _accum += _Query_val_ptr[q] * _Key_val_ptr[q];\n"
    "    }\n"
    "    _accum *= _inv_17;\n"
    "    Sampled_values[pMask0] = _accum;\n"
)
_COMPRESSED_MASK_HOISTED_LOOP = (
    "      float _accum = 0.0f;\n"
    "       \n"
    "      const float* __restrict__ _Key_val_ptr = &Key_val[pKey0 * Key1_size];\n"
    "      const float* __restrict__ _Query_val_ptr = "
    "&Query_val[pQuery0 * Query1_size];\n"
    "      float _inv_11 = Mask_val[pMask1];\n"
    "      #pragma omp simd\n"
    "      for (int64_t q = 0; q < Query1_size; q++) {\n"
    "        // Resolve dense coordinates\n"
    "        _accum += _Query_val_ptr[q] * _Key_val_ptr[q];\n"
    "      }\n"
    "      _accum *= _inv_11;\n"
    "      Sampled_values[pSampled1] = _accum;\n"
)


@pytest.mark.parametrize("regblock_enabled", (False, True), ids=("off", "on"))
@pytest.mark.parametrize(
    ("mask_format", "hoisted_loop"),
    (
        ("oo", _COORDINATE_MASK_HOISTED_LOOP),
        ("ds", _COMPRESSED_MASK_HOISTED_LOOP),
    ),
)
def test_float32_sddmm_kernel_keeps_its_hoisted_loop_byte_for_byte(
    mask_format: str, hoisted_loop: str, regblock_enabled: bool
) -> None:
    with regblock_force(regblock_enabled):
        lowered = CINLowerer().lower_IndexStmt(
            _all_coordinate_result_sddmm(mask_format)
        )
        cpp = LLIRLowerer().lower_llir(lowered)

    assert cpp.count(hoisted_loop) == 1
    assert cpp.count("_inv_") == 2
    # The coordinate-mask kernel runs its loop over the mask in parallel.  Before patch
    # 0047 it wrote the result at a shared counter (pSampled1++) inside that loop, a
    # data race; it now writes at the mask entry's own position (pMask0).  No kernel
    # here may write a scalar that the parallel loop shares.
    assert shared_scalar_writes(cpp) == []


def _options(regblock_enabled: bool) -> CompileOptions:
    return replace(
        CompileOptions.from_environment().with_regblock_enabled(regblock_enabled),
        requested_schedule=Schedule(),
    )


# -- valid: a matrix-valued reduction on the LoopIR route (nothing compiled) ----


def _matrix_valued_reduction(tensor_format: str, result_format: str) -> ForAll:
    """``C[i, j] += A[i, j, k] * B[j] * D[k]``: one accumulator per ``(i, j)``."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    tensor = TensorVar("A", fmt=tensor_format, dtype=torch.float64)[i, j, k]
    middle = TensorVar("B", fmt="d", dtype=torch.float64)[j]
    inner = TensorVar("D", fmt="d", dtype=torch.float64)[k]
    product = CINBinaryOp(
        Operation.MUL, CINBinaryOp(Operation.MUL, tensor, middle), inner
    )
    result = TensorVar("C", fmt=result_format, dtype=torch.float64)[i, j]
    return ForAll(
        i, ForAll(j, ForAll(k, TensorAssign(result, product, op=Operation.ADD)))
    )


@pytest.mark.parametrize("regblock_enabled", (False, True), ids=("off", "on"))
@pytest.mark.parametrize(
    ("tensor_format", "result_format"),
    (("sss", "ss"), ("sss", "ds"), ("dss", "ss"), ("sds", "ss")),
)
def test_reduction_zeroed_directly_before_its_loop_still_hoists(
    tensor_format: str, result_format: str, regblock_enabled: bool
) -> None:
    extents_i, extents_j, extents_k = EXTENTS
    kernel = compile_cin_via_loopir(
        _matrix_valued_reduction(tensor_format, result_format),
        (extents_i, extents_j),
        (
            (EXTENTS, torch.float64),
            ((extents_j,), torch.float64),
            ((extents_k,), torch.float64),
        ),
        compile_options=_options(regblock_enabled),
    )
    cpp = kernel.cpp_source

    # Here the same ``B[j]`` is hoisted out of the same ``k`` loop, and validly:
    # the accumulator belongs to one ``(i, j)`` and is zeroed inside the ``j`` loop.
    hoisted = re.search(
        r"^( +)double C_reduction = 0\.0;\n"
        r"(?:\1[^\n]*\n)*?"
        r"\1double (_inv_\d+) = B_val\[j\];\n"
        r"\1for \([^\n]*\n"
        r"(?:\1  [^\n]*\n)*?"
        r"\1  C_reduction \+= [^\n]*D_val\[k\];\n"
        r"(?:\1  [^\n]*\n)*?"
        r"\1}\n"
        r"\1C_reduction \*= \2;\n",
        cpp,
        flags=re.MULTILINE,
    )
    assert hoisted is not None
    assert cpp.count("_inv_") == 2
    assert not re.search(r"C_reduction \+= .*B_val", cpp)


# -- not valid: the three-factor reduction keeps its factor (nothing compiled) --


def _three_factor_reduction(dtype: torch.dtype) -> ForAll:
    """``C[i] += A[i, j, k] * B[j] * D[k]`` into a compressed vector."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    tensor = TensorVar("A", fmt="sss", dtype=dtype)[i, j, k]
    middle = TensorVar("B", fmt="d", dtype=dtype)[j]
    inner = TensorVar("D", fmt="d", dtype=dtype)[k]
    product = CINBinaryOp(
        Operation.MUL, CINBinaryOp(Operation.MUL, tensor, middle), inner
    )
    result = TensorVar("C", fmt="s", dtype=dtype)[i]
    return ForAll(
        i, ForAll(j, ForAll(k, TensorAssign(result, product, op=Operation.ADD)))
    )


# ``C_nonempty``: the result stores ``i`` only when some entry under row ``i``
# contributes, so a row whose stored ``j`` segments are all empty stores nothing.
_THREE_FACTOR_REDUCTION_NEST = (
    "    // Initialize the reduction accumulator\n"
    "    double C_reduction = 0.0;\n"
    "    bool C_nonempty = false;\n"
    "     \n"
    "    // Initialize iterators\n"
    "    int pA1_end = A1_pos[pA0 + 1];\n"
    "     \n"
    "    for (int pA1 = A1_pos[pA0]; pA1 < pA1_end; pA1++) {\n"
    "      // Resolve coordinates\n"
    "      int j = A1_crd[pA1];\n"
    "       \n"
    "      // Resolve dense coordinates\n"
    "      int pB0 = j;\n"
    "      // Initialize iterators\n"
    "      int pA2_end = A2_pos[pA1 + 1];\n"
    "       \n"
    "      for (int pA2 = A2_pos[pA1]; pA2 < pA2_end; pA2++) {\n"
    "        // Resolve coordinates\n"
    "        int k = A2_crd[pA2];\n"
    "         \n"
    "        // Resolve dense coordinates\n"
    "        int pD0 = k;\n"
    "        C_reduction += A_val[pA2] * B_val[j] * D_val[k];\n"
    "        C_nonempty = true;\n"
    "      }\n"
    "    }\n"
    "     \n"
    "    if (C_nonempty) {\n"
    "      C_values.emplace_back(C_reduction);\n"
)


@pytest.mark.parametrize("regblock_enabled", (False, True), ids=("off", "on"))
@pytest.mark.parametrize(
    ("dtype", "scalar"),
    ((torch.float64, "double"), (torch.float32, "float")),
    ids=("float64", "float32"),
)
def test_three_factor_reduction_keeps_the_invariant_factor_inside_the_loop(
    dtype: torch.dtype, scalar: str, regblock_enabled: bool
) -> None:
    extents_i, extents_j, extents_k = EXTENTS
    kernel = compile_cin_via_loopir(
        _three_factor_reduction(dtype),
        (extents_i,),
        ((EXTENTS, dtype), ((extents_j,), dtype), ((extents_k,), dtype)),
        compile_options=_options(regblock_enabled),
    )
    cpp = kernel.cpp_source

    # The accumulator is zeroed outside the ``j`` loop, so it is not zero on
    # entry to the ``k`` loop and nothing may be hoisted out of that loop.
    assert "_inv_" not in cpp
    assert "C_reduction *=" not in cpp
    assert cpp.count(_THREE_FACTOR_REDUCTION_NEST.replace("double", scalar)) == 1


# -- not valid: the three-factor reduction's values (compiles kernels) ---------


@pytest.mark.parametrize("regblock_enabled", (False, True), ids=("off", "on"))
@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    (
        (torch.float64, FLOAT64_RELATIVE_TOLERANCE),
        (torch.float32, FLOAT32_RELATIVE_TOLERANCE),
    ),
    ids=("float64", "float32"),
)
def test_three_factor_reduction_matches_float64_einsum(
    dtype: torch.dtype, tolerance: float, regblock_enabled: bool
) -> None:
    generator = torch.Generator().manual_seed(11)
    extents_i, extents_j, extents_k = EXTENTS
    dense = (torch.rand(EXTENTS, generator=generator, dtype=torch.float64) + 0.5) * (
        torch.rand(EXTENTS, generator=generator) < 0.6
    )
    middle = torch.rand((extents_j,), generator=generator, dtype=torch.float64) + 0.5
    inner = torch.rand((extents_k,), generator=generator, dtype=torch.float64) + 0.5
    # Every ``i`` stores more than one ``j``: one ``j`` alone would hide a
    # factor that rescales the sums of the ``j`` before it.
    assert bool(((dense != 0).any(dim=2).sum(dim=1) >= 2).all())
    expected = torch.einsum(
        "ijk,j,k->i",
        dense.to(dtype).double(),
        middle.to(dtype).double(),
        inner.to(dtype).double(),
    )

    result, _ = execute_cin_via_loopir(
        _three_factor_reduction(dtype),
        (extents_i,),
        STensor.from_torch(dense.to(dtype).clone(), "A").to_sparse("sss"),
        STensor.from_torch(middle.to(dtype).clone(), "B").to_dense(),
        STensor.from_torch(inner.to(dtype).clone(), "D").to_dense(),
        compile_options=_options(regblock_enabled),
    )

    actual = result.to_torch(in_place=False).double()
    assert actual.shape == expected.shape
    assert float(((actual - expected).abs() / expected.abs()).max()) <= tolerance


# -- not valid: a factor of the outer of two reduction loops -------------------
#
# From the independent review of a20fe61 (/Users/bobby/loopir-review, finding
# 1).  The first test compiles nothing; the second compiles kernels.


def _dense_row_two_loop_reduction(dtype: torch.dtype) -> ForAll:
    """``C[i] += A[i, k, r] * B[i, k]``, ``A`` dense then compressed twice, into a compressed vector."""

    i, k, r = IndexVar("i"), IndexVar("k"), IndexVar("r")
    tensor = TensorVar("A", fmt="dss", dtype=dtype)[i, k, r]
    outer = TensorVar("B", fmt="dd", dtype=dtype)[i, k]
    result = TensorVar("C", fmt="s", dtype=dtype)[i]
    product = CINBinaryOp(Operation.MUL, tensor, outer)
    return ForAll(
        i, ForAll(k, ForAll(r, TensorAssign(result, product, op=Operation.ADD)))
    )


@pytest.mark.parametrize("regblock_enabled", (False, True), ids=("off", "on"))
@pytest.mark.parametrize(
    "dtype", (torch.float64, torch.float32), ids=("float64", "float32")
)
def test_two_loop_reduction_keeps_the_outer_loops_factor_inside_the_inner_loop(
    dtype: torch.dtype, regblock_enabled: bool
) -> None:
    kernel = compile_cin_via_loopir(
        _dense_row_two_loop_reduction(dtype),
        (1,),
        (((1, 2, 2), dtype), ((1, 2), dtype)),
        compile_options=_options(regblock_enabled),
    )
    cpp = kernel.cpp_source

    # The accumulator is zeroed outside the ``k`` loop, so it is not zero on
    # entry to the ``r`` loop and ``B[i, k]`` may not be hoisted out of it.
    assert "_inv_" not in cpp
    assert "C_reduction *=" not in cpp
    assert cpp.count("        C_reduction += A_val[pA2] * B_val[pB1];\n") == 1


@pytest.mark.parametrize("regblock_enabled", (False, True), ids=("off", "on"))
@pytest.mark.parametrize(
    "index_dtype", (torch.int32, torch.int64), ids=("int32", "int64")
)
@pytest.mark.parametrize(
    "dtype", (torch.float64, torch.float32), ids=("float64", "float32")
)
def test_two_loop_reduction_returns_ten_not_eighteen(
    dtype: torch.dtype, index_dtype: torch.dtype, regblock_enabled: bool
) -> None:
    def index_array(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=index_dtype)

    tensor = STensor.from_components(
        (1, 2, 2),
        "dss",
        [
            [],
            [index_array([0, 2]), index_array([0, 1])],
            [index_array([0, 2, 4]), index_array([0, 1, 0, 1])],
        ],
        torch.tensor([1, 1, 1, 1], dtype=dtype),
        name="A",
    )
    outer = STensor.from_components(
        (1, 2), "dd", [[], []], torch.tensor([2, 3], dtype=dtype), name="B"
    )

    result, _ = execute_cin_via_loopir(
        _dense_row_two_loop_reduction(dtype),
        (1,),
        tensor,
        outer,
        compile_options=_options(regblock_enabled),
    )

    # (1 + 1) * 2 + (1 + 1) * 3.  Rescaling the first term by 3 gives 18.
    position, coordinate = result.storage.index.mode_indices[0]
    assert position.tolist() == [0, 1]
    assert coordinate.tolist() == [0]
    assert result.storage.value.tolist() == [10.0]
