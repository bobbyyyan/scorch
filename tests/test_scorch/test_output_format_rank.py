"""A requested output format whose rank is not the result's, refused at the front door (patch 0072).

``scorch.matmul``'s vector branch and ``scorch.ops.spmv`` took ``format=`` / ``output_format=``
without checking it against the rank-1 result.  The CIN they built from a rank-2 request then
failed the compiler's verifier at a stage boundary (``invalid_cin_field``), a code that names
an internal invariant rather than the caller's mistake; before the merge (864745d) the same
call crashed the process.  ``einsum`` already refused the mistake with
``einsum_output_format_rank_mismatch``.  Now ``einsum``, both branches of ``matmul`` and
``spmv`` refuse it with that code, from one check (``ops._require_output_format_rank``), before
any CIN exists.

1. The vector branch: every combination of operand (STensor, torch CSR), dtype, index width,
   ``requires_grad`` and keyword (``format``, ``output_format``) with a rank-2 request is
   refused with the code and builds nothing; the same call without the keyword computes
   ``A @ v`` (float64 reference).
2. ``scorch.ops.spmv`` with an output format of rank 0, 2 and 3.
3. A population of output-format mistakes through ``matmul``, ``spmv`` and ``einsum``: each is
   refused with the code of its cause (rank, unknown level letter, wrong type), builds nothing,
   and no exception in its chain is the compiler's ``VerificationError``.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import pytest
import torch

import scorch
from scorch import ops
from scorch.compiler.diagnostics import VerificationError
from scorch.exceptions import defect_code
from scorch.stensor import STensor

RANK = "einsum_output_format_rank_mismatch"
ROWS, COLS, WIDTH = 7, 5, 3


@pytest.fixture
def builds(monkeypatch) -> List[int]:
    """Counts the builds of generated kernels; a build still runs."""
    count: List[int] = []
    build = ops._prepare_generated_kernel_build

    def counted(*args, **kwargs):
        count.append(1)
        return build(*args, **kwargs)

    monkeypatch.setattr(ops, "_prepare_generated_kernel_build", counted)
    return count


def _csr(dtype: torch.dtype, index: torch.dtype) -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    dense = (torch.rand(ROWS, COLS, generator=g) < 0.4) * torch.rand(ROWS, COLS, generator=g)
    dense[0, 0] = 1.0  # at least one stored entry
    csr = dense.to(dtype).to_sparse_csr()
    return torch.sparse_csr_tensor(
        csr.crow_indices().to(index),
        csr.col_indices().to(index),
        csr.values(),
        size=(ROWS, COLS),
    )


def _chain(error: BaseException) -> List[BaseException]:
    seen, out, e = set(), [], error
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        out.append(e)
        e = e.__cause__ or e.__context__
    return out


def _assert_refused(call: Callable[[], object], code: str, builds: List[int]) -> None:
    before = len(builds)
    with pytest.raises(Exception) as caught:
        call()
    assert defect_code(caught.value) == code, repr(caught.value)
    assert not any(isinstance(e, VerificationError) for e in _chain(caught.value))
    assert len(builds) == before, "a refused call built a kernel"


@pytest.mark.parametrize("keyword", ["format", "output_format"])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("index", [torch.int32, torch.int64], ids=["int32", "int64"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
@pytest.mark.parametrize("kind", ["stensor", "torch_csr"])
def test_vector_branch_refuses_a_rank_2_output_format(
    kind, dtype, index, requires_grad, keyword, builds
):
    csr = _csr(dtype, index)
    a = STensor.from_csr(csr, "A") if kind == "stensor" else csr
    v = torch.rand(COLS, dtype=dtype, generator=torch.Generator().manual_seed(7))
    v.requires_grad_(requires_grad)

    _assert_refused(lambda: scorch.matmul(a, v, **{keyword: "dd"}), RANK, builds)

    y = scorch.matmul(a, v)
    reference = csr.to_dense().double() @ v.detach().double()
    assert tuple(y.shape) == (ROWS,)
    tol = 1e-6 if dtype == torch.float32 else 1e-13
    assert torch.allclose(y.detach().double(), reference, rtol=tol, atol=tol)


@pytest.mark.parametrize("requested", ["", "dd", "ss", "ddd"])
def test_spmv_refuses_an_output_format_of_another_rank(requested, builds):
    a = STensor.from_csr(_csr(torch.float32, torch.int64), "A")
    x = STensor.from_torch(torch.rand(COLS))
    _assert_refused(lambda: ops.spmv(a, x, output_format=requested), RANK, builds)


def _population() -> List[Tuple[str, Callable[[], object], str]]:
    csr = _csr(torch.float32, torch.int64)
    st = STensor.from_csr(csr, "A")
    v, x = torch.rand(COLS), torch.rand(COLS, WIDTH)
    left, lv = torch.rand(WIDTH, ROWS), torch.rand(ROWS)
    vs = STensor.from_torch(v)
    cases: List[Tuple[str, Callable[[], object], str]] = []
    # (requested format, the code for a rank-1 result, the code for a rank-2 result)
    for requested, code1, code2 in [
        ("", RANK, RANK),
        ("s", None, RANK),
        ("dd", RANK, None),
        ("ss", RANK, None),
        ("ddd", RANK, RANK),
        ("x", "level_format_unknown", "level_format_unknown"),
        (["dense", "bogus"], "level_format_unknown", "level_format_unknown"),
        (7, "tensor_format_type_invalid", "tensor_format_type_invalid"),
    ]:
        r = repr(requested)
        for name, a in (("stensor", st), ("torch_csr", csr)):
            if code1 is not None:
                for kw in ("format", "output_format"):
                    cases.append(
                        (f"matmul({name}, v, {kw}={r})",
                         lambda a=a, kw=kw, f=requested: scorch.matmul(a, v, **{kw: f}),
                         code1)
                    )
                cases.append(
                    (f"matmul(v, {name}, format={r})",
                     lambda a=a, f=requested: scorch.matmul(lv, a, format=f),
                     code1)
                )
            if code2 is not None:
                cases.append(
                    (f"matmul({name}, X, format={r})",
                     lambda a=a, f=requested: scorch.matmul(a, x, format=f),
                     code2)
                )
                cases.append(
                    (f"matmul(X, {name}, format={r})",
                     lambda a=a, f=requested: scorch.matmul(left, a, format=f),
                     code2)
                )
        if code1 is not None:
            cases.append(
                (f"spmv(stensor, v, output_format={r})",
                 lambda f=requested: ops.spmv(st, vs, output_format=f),
                 code1)
            )
            cases.append(
                (f"einsum('ij,j->i', format={r})",
                 lambda f=requested: scorch.einsum("ij,j->i", st, v, format=f),
                 code1)
            )
        if code2 is not None:
            cases.append(
                (f"einsum('ij,jk->ik', format={r})",
                 lambda f=requested: scorch.einsum("ij,jk->ik", st, x, format=f),
                 code2)
            )
    return cases


_POPULATION = _population()


@pytest.mark.parametrize(
    "call, code", [c[1:] for c in _POPULATION], ids=[c[0] for c in _POPULATION]
)
def test_an_output_format_mistake_is_refused_with_its_own_code(call, code, builds):
    _assert_refused(call, code, builds)
