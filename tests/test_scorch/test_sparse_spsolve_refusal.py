"""scorch.sparse: torch.sparse's namespace, with spsolve refused by code where no kernel
solves the system (POST_LANDING item 81 (j)).

``import scorch as torch`` makes ``torch.sparse`` scorch's module, which hands back
torch.sparse's own object for every name but spsolve.  Scorch has no sparse solver, so
spsolve is torch's call where torch's dispatcher holds a kernel for the operands'
backend (a CSR matrix on a CUDA build with cuDSS) and a coded refusal elsewhere: an
STensor operand always, and on CPU every torch layout.  Before, an STensor reached
torch's argument parser (uncoded TypeError) and a torch layout reached torch's
dispatcher (uncoded "Could not run 'aten::_spsolve'").
"""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import scorch
import scorch.sparse
from scorch.exceptions import (
    SparseSolveUnsupportedError,
    TensorTypeError,
    TensorValidationError,
    defect_code,
)

N = 6

# The torch layouts of a square matrix, as the item 81 probe built them.
LAYOUTS = {
    "coo": lambda d: d.to_sparse(),
    "hybrid_coo": lambda d: d.to_sparse(sparse_dim=1),
    "csr": lambda d: d.to_sparse_csr(),
    "csc": lambda d: d.to_sparse_csc(),
    "bsr": lambda d: d.to_sparse_bsr((2, 2)),
    "bsc": lambda d: d.to_sparse_bsc((2, 2)),
}
DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int64]


def _system(dtype):
    """A dense N x N matrix with a dominant diagonal, and a right-hand side."""

    generator = torch.Generator().manual_seed(81)
    a = torch.rand(N, N, generator=generator, dtype=torch.float64)
    a = torch.where(a < 0.5, torch.zeros_like(a), a) + 16 * torch.eye(N)
    b = torch.rand(N, generator=generator, dtype=torch.float64)
    return a.to(dtype), b.to(dtype)


def _refused(call):
    with pytest.raises(SparseSolveUnsupportedError) as caught:
        call()
    assert defect_code(caught.value) == "spsolve_unsupported"
    assert isinstance(caught.value, NotImplementedError)
    return caught.value


def test_scorch_sparse_is_torch_sparse_but_spsolve():
    assert scorch.sparse is not torch.sparse
    names = [n for n in dir(torch.sparse) if not n.startswith("_") and n != "spsolve"]
    assert "mm" in names and "addmm" in names and "softmax" in names
    for name in names:
        assert getattr(scorch.sparse, name) is getattr(torch.sparse, name), name
    assert scorch.sparse.spsolve is not torch.sparse.spsolve
    assert set(names) <= set(dir(scorch.sparse))


def test_star_import_binds_scorchs_spsolve_and_torchs_other_names():
    namespace: dict = {}
    exec("from scorch.sparse import *", namespace)
    assert namespace["spsolve"] is scorch.sparse.spsolve
    assert namespace["mm"] is torch.sparse.mm
    assert "solve" not in namespace  # torch.sparse.__all__ lists it, defines nothing


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_torch_layouts_are_refused_where_torch_has_no_kernel(layout, dtype):
    """Torch's own call says which branch the case is in: where it raises
    NotImplementedError (no kernel; every CPU layout), scorch refuses by code, and where
    it computes, scorch returns torch's result."""

    dense, rhs = _system(dtype)
    matrix = LAYOUTS[layout](dense)
    try:
        expected = torch.sparse.spsolve(matrix, rhs)
    except NotImplementedError:
        error = _refused(lambda: scorch.sparse.spsolve(matrix, rhs))
        assert "no spsolve kernel" in str(error)
        _refused(lambda: scorch.sparse.spsolve(matrix, rhs, left=False))
        return
    torch.testing.assert_close(scorch.sparse.spsolve(matrix, rhs), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64], ids=str)
@pytest.mark.parametrize("layout", ["csr", "coo"])
def test_an_stensor_operand_is_refused(layout, dtype):
    dense, rhs = _system(dtype)
    matrix = scorch.from_torch(LAYOUTS[layout](dense))
    error = _refused(lambda: scorch.sparse.spsolve(matrix, rhs))
    assert "input is an STensor" in str(error)
    error = _refused(
        lambda: scorch.sparse.spsolve(LAYOUTS[layout](dense), scorch.from_torch(rhs))
    )
    assert "other is an STensor" in str(error)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("layout", ["csr", "coo"])
def test_a_16_bit_stensor_is_still_refused_when_it_is_built(layout, dtype):
    """A float16 or bfloat16 STensor is refused where it is constructed, before any
    spsolve call, as before (dtype_unsupported)."""

    dense, _ = _system(dtype)
    with pytest.raises(TensorValidationError) as caught:
        scorch.from_torch(LAYOUTS[layout](dense))
    assert defect_code(caught.value) == "dtype_unsupported"


@pytest.mark.parametrize("bad", [[[1.0]], 2.0, None], ids=["list", "float", "none"])
def test_an_operand_that_is_not_a_tensor_is_refused(bad):
    dense, rhs = _system(torch.float64)
    for call in (
        lambda: scorch.sparse.spsolve(bad, rhs),
        lambda: scorch.sparse.spsolve(dense.to_sparse_csr(), bad),
    ):
        with pytest.raises(TensorTypeError) as caught:
            call()
        assert defect_code(caught.value) == "spsolve_operand_type_invalid"
        assert isinstance(caught.value, TypeError)


def test_the_backend_keys_lie_below_backend_select():
    """spsolve reads the backend key as the highest key below BackendSelect.  On this
    build the keys between ADInplaceOrView and BackendSelect (ZeroTensor, Negative,
    Conjugate, Functionalize, Python) sit above that set, and the backend keys in it."""

    keys = scorch.sparse._BACKEND_KEYS
    above = ["ADInplaceOrView", "ZeroTensor", "Negative", "Conjugate", "Functionalize"]
    for name in above + ["Python", "BackendSelect"]:
        assert not keys.has(getattr(torch._C.DispatchKey, name)), name
    for name in ("CPU", "SparseCPU", "SparseCsrCPU", "CUDA", "SparseCsrCUDA"):
        assert keys.has(getattr(torch._C.DispatchKey, name)), name


def test_a_conjugated_operand_is_judged_by_its_backend():
    """A conjugated right-hand side carries the Conjugate key, whose fallback
    redispatches; the call's backend is still the CSR matrix's.  torch names that
    backend in its own error, and so does the refusal."""

    dense, rhs = _system(torch.complex64)
    matrix, conjugated = dense.to_sparse_csr(), (rhs * (1 + 2j)).conj()
    assert conjugated.is_conj()
    assert torch._C._dispatch_keys(conjugated).has(torch._C.DispatchKey.Conjugate)
    with pytest.raises(NotImplementedError, match="SparseCsrCPU"):
        torch.sparse.spsolve(matrix, conjugated)
    error = _refused(lambda: scorch.sparse.spsolve(matrix, conjugated))
    assert "SparseCsrCPU" in str(error)


def test_a_torch_function_override_takes_the_call():
    """An operand that defines __torch_function__ gets the call, as torch's protocol
    gives it: scorch hands it to torch.sparse.spsolve unchanged."""

    calls = []

    class Override(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            calls.append(func)
            return "taken by __torch_function__"

    dense, rhs = _system(torch.float64)
    got = scorch.sparse.spsolve(dense.to_sparse_csr(), rhs.as_subclass(Override))
    assert got == "taken by __torch_function__"
    assert calls == [torch.sparse.spsolve]


def test_a_dispatch_mode_takes_the_call():
    """Under a TorchDispatchMode the thread's dispatch key set holds the Python key and
    torch's mode gets the call (has_torch_function is False there)."""

    seen = []

    class Mode(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(str(func))
            return torch.zeros(N, dtype=torch.float64)

    dense, rhs = _system(torch.float64)
    matrix = dense.to_sparse_csr()
    with Mode():
        got = scorch.sparse.spsolve(matrix, rhs)
    assert seen == ["aten._spsolve.default"]
    torch.testing.assert_close(got, torch.zeros(N, dtype=torch.float64))


def test_a_torch_dispatch_subclass_takes_the_call():
    """An operand that defines __torch_dispatch__ carries the Python key in its own key
    set, and its handler gets the call."""

    seen = []

    class Wrapper(torch.Tensor):
        @staticmethod
        def __new__(cls, elem):
            return torch.Tensor._make_wrapper_subclass(
                cls, elem.shape, dtype=elem.dtype, device=elem.device
            )

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            seen.append(str(func))
            return torch.zeros(N, dtype=torch.float64)

        __torch_function__ = torch._C._disabled_torch_function_impl

    dense, rhs = _system(torch.float64)
    wrapped = Wrapper(rhs)
    assert torch._C._dispatch_keys(wrapped).has(torch._C.DispatchKey.Python)
    got = scorch.sparse.spsolve(dense.to_sparse_csr(), wrapped)
    assert seen == ["aten._spsolve.default"]
    torch.testing.assert_close(got, torch.zeros(N, dtype=torch.float64))


def test_a_backend_with_a_kernel_is_torchs_call(monkeypatch):
    """Where torch's dispatcher holds an aten::_spsolve kernel for the operands'
    backend, the call is torch.sparse.spsolve's, unchanged.  No CPU build holds one, so
    here the dispatcher's answer is replaced by one that holds a kernel for SparseCsrCPU
    only, and torch.sparse.spsolve by a recorder; the CUDA test below makes the real
    call.  A COO matrix, whose backend is SparseCPU, is still refused."""

    def has_kernel(operator, key):
        return operator == "aten::_spsolve" and key == torch._C.DispatchKey.SparseCsrCPU

    calls = []
    answer = torch.ones(N, dtype=torch.float64)

    def recorder(input, other, *, left=True):
        calls.append((input, other, left))
        return answer

    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", has_kernel)
    monkeypatch.setattr(torch.sparse, "spsolve", recorder)
    dense, rhs = _system(torch.float64)
    matrix = dense.to_sparse_csr()
    assert scorch.sparse.spsolve(matrix, rhs, left=False) is answer
    assert len(calls) == 1
    assert calls[0][0] is matrix and calls[0][1] is rhs and calls[0][2] is False
    error = _refused(lambda: scorch.sparse.spsolve(dense.to_sparse(), rhs))
    assert "SparseCPU" in str(error)
    assert len(calls) == 1


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::_spsolve", "SparseCsrCUDA"
    ),
    reason="this torch build has no CUDA spsolve kernel (cuDSS)",
)
def test_a_cuda_csr_system_is_solved_by_torch():
    dense, rhs = _system(torch.float64)
    matrix, rhs = dense.to_sparse_csr().cuda(), rhs.cuda()
    got = scorch.sparse.spsolve(matrix, rhs)
    torch.testing.assert_close(got, torch.sparse.spsolve(matrix, rhs))
    torch.testing.assert_close(got, torch.linalg.solve(dense.cuda(), rhs))
