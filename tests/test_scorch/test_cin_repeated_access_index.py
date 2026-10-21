"""CIN verification refuses an access that repeats an index variable (patch 0081).

``A[i, i]`` reads A only where its two coordinates are equal (its diagonal).  The shipped
compiler has no lowering for that: its value read resolves at the first occurrence's
level, so ``B[i] = A[i, i]`` read ``A_val[pA0]`` with ``pA0 = i`` (COMPILER_IR_REFACTOR_
PHASE4_REVIEW.md section 4, item 2).  On the shipped line before the merge (864745d) an
unrelated failure later in ``lower_and_exec_cin`` raised before those values reached the
caller; once that failure was fixed they came back silently (probe jobs 83947/83948,
2026-10-14).  ``B[i, j] = A[i, i, j]`` stored dense was wrong on both.

The structural check that every compiler entry runs on its CIN, in release mode too
(``cin_analysis.verify_cin_structure``, called by ``normalize_cin``), now refuses such an
access with ``unsupported_repeated_access_index``, the new compiler's code for it, as a
``VerificationError`` whose diagnostics carry the code (``defect_code`` reads it).  It
refuses before lowering, before any kernel is built and before any kernel cache is
consulted, so a kernel compiled by an earlier process cannot serve such a CIN either.
``scorch.einsum`` refuses the spelling earlier, in its own label check
(``einsum_repeated_index``); ``scorch.ops.lower_and_exec_cin`` is the public entry that
takes a caller's CIN.
"""

import pytest
import torch
import torch.utils.cpp_extension

import scorch.ops as ops
import scorch.utils as utils
from scorch.compiler.cin import ForAll, IndexVar, TensorAssign, TensorVar
from scorch.compiler.cin_analysis import normalize_cin
from scorch.compiler.compile_options import CompileOptions
from scorch.compiler.diagnostics import VerificationError
from scorch.exceptions import EinsumRepeatedIndexError, defect_code
from scorch.stensor import STensor

CODE = "unsupported_repeated_access_index"
N = 5

# (statement, operand format, result format): the ten cases of the probe that found it.
CASES = [
    ("ii->i", "dd", "d"),
    ("ii->i", "ds", "d"),
    ("ii->i", "sd", "d"),
    ("ii->i", "ss", "d"),
    ("ii->i", "oo", "d"),
    ("ii->i", "ds", "s"),
    ("iij->ij", "ddd", "dd"),
    ("iij->ij", "dss", "dd"),
    ("iij->ij", "sss", "dd"),
    ("iij->ij", "dds", "ds"),
]


def _operand(fmt_a):
    """The probe's operand: random off the diagonal, distinct values on it."""

    rank = len(fmt_a)
    shape = (N, N) if rank == 2 else (N, N, 5)
    generator = torch.Generator().manual_seed(0)
    keep = torch.rand(shape, generator=generator) < 0.6
    dense = torch.rand(shape, generator=generator, dtype=torch.float64) * keep
    for d in range(N):
        if rank == 2:
            dense[d, d] = 1.0 + d
        else:
            dense[d, d, :] = 1.0 + d + torch.arange(5, dtype=torch.float64) / 8
    a = STensor.from_torch(dense, "A")
    return a.to_dense() if set(fmt_a) == {"d"} else a.to_sparse(fmt_a)


def _statement(expression, fmt_a, out_fmt):
    i, j = IndexVar("i"), IndexVar("j")
    rank = len(fmt_a)
    a = TensorVar("A", fmt=fmt_a, mode_order=list(range(rank)), dtype=torch.float64)
    if expression == "ii->i":
        out = TensorVar("B", fmt=out_fmt, mode_order=[0], dtype=torch.float64)
        return ForAll(i, TensorAssign(out[i], a[i, i])), (N,)
    out = TensorVar("B", fmt=out_fmt, mode_order=[0, 1], dtype=torch.float64)
    return ForAll(i, ForAll(j, TensorAssign(out[i, j], a[i, i, j]))), (N, 5)


def _forbid(monkeypatch, target, name):
    def refuse(*args, **kwargs):
        raise AssertionError(f"{name} ran for a CIN that verification must refuse")

    monkeypatch.setattr(target, name, refuse)


@pytest.mark.parametrize("expression,fmt_a,out_fmt", CASES)
def test_lower_and_exec_cin_refuses_before_any_later_step(
    monkeypatch, expression, fmt_a, out_fmt
):
    """The refusal comes from verification, and nothing after it runs.

    The step that follows normalization in ``lower_and_exec_cin`` (binding the runtime
    operands), the kernel loader (which also finds kernels an earlier process built) and
    torch's inline build all fail the test if reached.
    """

    a = _operand(fmt_a)
    stmt, result_shape = _statement(expression, fmt_a, out_fmt)
    _forbid(monkeypatch, ops, "_plan_direct_cin_runtime_binding")
    _forbid(monkeypatch, utils, "_load_kernel")
    _forbid(monkeypatch, torch.utils.cpp_extension, "load_inline")
    with pytest.raises(VerificationError) as raised:
        ops.lower_and_exec_cin(stmt, result_shape, a)
    assert defect_code(raised.value) == CODE
    assert raised.value.diagnostics[0].code == CODE


@pytest.mark.parametrize("verify_cin", [False, True])
def test_normalize_cin_refuses_in_release_and_debug_mode(verify_cin):
    """The check is structural, so it runs whether or not full verification is on."""

    stmt, _ = _statement("ii->i", "ds", "d")
    options = CompileOptions.from_environment(
        environ={}, verify_cin_override=verify_cin
    )
    assert options.verification.verify_cin is verify_cin
    with pytest.raises(VerificationError) as raised:
        normalize_cin(stmt, compile_options=options)
    assert defect_code(raised.value) == CODE


def test_the_diagnostic_names_the_second_occurrence():
    i = IndexVar("i")
    a = TensorVar("A", fmt="sss", dtype=torch.float64)
    out = TensorVar("B", fmt="d", dtype=torch.float64)
    stmt = ForAll(i, TensorAssign(out[i], a[i, i, i]))
    with pytest.raises(VerificationError) as raised:
        normalize_cin(stmt)
    first = raised.value.diagnostics[0]
    assert first.code == CODE
    assert first.path[-2:] == ("index_ids", "[1]")
    assert "axes 0 and 1" in first.message
    # One diagnostic per access, however many times the variable repeats.
    assert [d.code for d in raised.value.diagnostics].count(CODE) == 1


def test_a_result_access_that_repeats_an_index_is_refused():
    i = IndexVar("i")
    a = TensorVar("A", fmt="d", dtype=torch.float64)
    out = TensorVar("C", fmt="dd", dtype=torch.float64)
    with pytest.raises(VerificationError) as raised:
        normalize_cin(ForAll(i, TensorAssign(out[i, i], a[i])))
    assert defect_code(raised.value) == CODE


def test_distinct_index_variables_still_verify():
    """Control: the same statements with distinct variables pass the check."""

    i, j, k = IndexVar("i"), IndexVar("j"), IndexVar("k")
    a2 = TensorVar("A", fmt="ds", dtype=torch.float64)
    a3 = TensorVar("A", fmt="dss", dtype=torch.float64)
    b = TensorVar("B", fmt="d", dtype=torch.float64)
    c = TensorVar("C", fmt="dd", dtype=torch.float64)
    normalize_cin(ForAll(i, ForAll(j, TensorAssign(b[i], a2[i, j]))))
    normalize_cin(ForAll(i, ForAll(j, ForAll(k, TensorAssign(c[i, k], a3[i, j, k])))))


def test_einsum_still_refuses_the_spelling_first():
    """einsum's label check comes before any CIN exists; its code is unchanged."""

    a = _operand("ds")
    with pytest.raises(EinsumRepeatedIndexError) as raised:
        ops.einsum("ii->i", a)
    assert raised.value.code == "einsum_repeated_index"
