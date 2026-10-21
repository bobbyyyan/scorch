"""Float64 products of a CSR matrix and a dense operand keep float64 precision
(series step 30).

The shipped performance line (864745d) computed ``einsum("ik,kj->ij")`` of a
float64 CSR matrix and a dense float64 matrix at float32 precision: every row was
off by 1e-7 to 4e-7, at 5 rows and at 1,024, whether the CSR operand came from a
torch CSR tensor or from ``to_sparse("ds")``, while ``matmul`` and the
matrix-vector ``einsum`` on the same operand were exact (sapling 85225,
2026-10-16).  The suite's usual tolerance (``assert_close``, 1e-3) cannot tell a
float32 result from a float64 one, so these tests compare each result with a
float64 reference to a relative 1e-12: in several result formats, both automatic
arms, both ways of making the CSR operand, and both sizes.  Every value is
positive, so no reference entry comes from cancellation and an elementwise
relative bound holds; row 1 is empty, so its entries must be exactly zero.

The judge fails a NaN or an infinity explicitly.  A control shows that it rejects
a planted NaN and the reference rounded through float32.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest
import torch

import scorch  # type: ignore[import-untyped]
from scorch import STensor  # type: ignore[import-untyped]
from scorch.compiler.compile_options import (  # type: ignore[import-untyped]
    CompileOptions,
)

F64 = torch.float64
RTOL = 1e-12
# rows, inner extent, columns of the dense matrix, density of the CSR matrix
SIZES = {"5 rows": (5, 5, 3, 0.5), "1024 rows": (1024, 1024, 16, 0.05)}
CONSTRUCTIONS = ["torch CSR", "to_sparse ds"]


def _set_arm(monkeypatch: pytest.MonkeyPatch, arm: int) -> None:
    monkeypatch.setenv("SCORCH_REGBLOCK", str(arm))
    assert CompileOptions.from_environment().scheduler.regblock_enabled == bool(arm)


def _operands(
    size: str, construction: str
) -> Tuple[STensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The CSR operand, its dense copy, a dense matrix and a vector."""
    rows, inner, columns, density = SIZES[size]
    generator = torch.Generator().manual_seed(30)
    a = torch.rand(rows, inner, generator=generator, dtype=F64) + 0.5
    a[torch.rand(rows, inner, generator=generator, dtype=F64) >= density] = 0.0
    a[1] = 0.0
    b = torch.rand(inner, columns, generator=generator, dtype=F64) + 0.5
    x = torch.rand(inner, generator=generator, dtype=F64) + 0.5
    if construction == "torch CSR":
        csr = STensor.from_torch(a.to_sparse_csr())
    else:
        csr = STensor.from_torch(a).to_sparse("ds")
    return csr, a, b, x


def _dense(result: Any) -> torch.Tensor:
    if isinstance(result, STensor):
        result = result.to_torch()
    return result if result.layout == torch.strided else result.to_dense()


def _judge(got: Any, want: torch.Tensor) -> List[str]:
    """What is wrong with ``got`` against the float64 reference ``want``."""
    got = _dense(got)
    failures = [] if got.dtype == F64 else [f"dtype {got.dtype}"]
    if tuple(got.shape) != tuple(want.shape):
        return failures + [f"shape {tuple(got.shape)}, want {tuple(want.shape)}"]
    got = got.to(F64)
    if not bool(torch.isfinite(got).all()):
        return failures + ["a NaN or an infinity"]
    difference = (got - want).abs()
    off = difference > RTOL * want.abs()
    if bool(off.any()):
        failures.append(
            f"{int(off.sum())} of {off.numel()} entries off by more than {RTOL} "
            f"relative; largest difference {float(difference.max()):.3e}"
        )
    return failures


def test_the_judge_rejects_a_nan_and_float32_precision() -> None:
    _, a, b, _ = _operands("5 rows", "torch CSR")
    want = a @ b
    assert _judge(want.clone(), want) == []
    planted = want.clone()
    planted[2, 1] = float("nan")
    assert _judge(planted, want) == ["a NaN or an infinity"]
    rounded = want.to(torch.float32).to(F64)
    assert _judge(rounded, want), "the reference rounded through float32 passed"


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
@pytest.mark.parametrize("size", list(SIZES))
@pytest.mark.parametrize("result_format", ["dd", "ds", "sd", "ss"])
def test_einsum_of_a_csr_and_a_dense_matrix(
    monkeypatch: pytest.MonkeyPatch,
    result_format: str,
    size: str,
    construction: str,
    arm: int,
) -> None:
    _set_arm(monkeypatch, arm)
    csr, a, b, _ = _operands(size, construction)
    got = scorch.einsum("ik,kj->ij", csr, b, format=result_format)
    failures = _judge(got, a @ b)
    assert not failures, failures


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
@pytest.mark.parametrize("size", list(SIZES))
@pytest.mark.parametrize("result_format", ["d", "s"])
def test_einsum_of_a_csr_matrix_and_a_vector(
    monkeypatch: pytest.MonkeyPatch,
    result_format: str,
    size: str,
    construction: str,
    arm: int,
) -> None:
    _set_arm(monkeypatch, arm)
    csr, a, _, x = _operands(size, construction)
    got = scorch.einsum("ik,k->i", csr, x, format=result_format)
    failures = _judge(got, a @ x)
    assert not failures, failures


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("construction", CONSTRUCTIONS)
@pytest.mark.parametrize("size", list(SIZES))
def test_matmul_of_a_csr_and_a_dense_matrix(
    monkeypatch: pytest.MonkeyPatch, size: str, construction: str, arm: int
) -> None:
    _set_arm(monkeypatch, arm)
    csr, a, b, _ = _operands(size, construction)
    failures = _judge(scorch.matmul(csr, b), a @ b)
    assert not failures, failures
