"""``use_cache`` means one thing at every product entry that takes it.

``matmul``, ``einsum`` and ``scorch.ops.spmv`` accept ``use_cache``.  ``False`` skips
the entry's prebuilt kernels, so the call gets the generated kernel; it does not
run the compiler again, since a kernel the caches already keep for the call still
serves it.  ``matmul`` has prebuilt SpMV, SpMM and SpGEMM kernels, ``einsum`` has
one (SDDMM, ``"ij,ik,jk->ij"`` with a float32 COO first operand and dense others),
and ``spmv`` has none: ``matmul`` tries the prebuilt CSR SpMV before it calls it.

Each test pairs the two values of the keyword.  A prebuilt kernel replaced by one
that raises shows which calls reach it: ``True`` must (else ``False`` would prove
nothing), ``False`` must not and must still compute the product.
"""

import pytest
import torch

import scorch
import scorch.ops as ops
from scorch import STensor


def _masked(shape, density, seed):
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(shape, generator=generator) + 0.25
    return values * (torch.rand(shape, generator=generator) < density)


def _refuse(name):
    def refuse(*args, **kwargs):
        raise AssertionError(f"the call reached {name}")

    return refuse


def _sddmm_operands():
    sampled = STensor.from_torch(_masked((16, 12), 0.2, 11), "S").to_sparse("oo")
    left = STensor.from_torch(_masked((16, 4), 1.0, 12), "A")
    right = STensor.from_torch(_masked((12, 4), 1.0, 13), "B")
    return sampled, left, right


def _sddmm_reference(sampled, left, right):
    mask = sampled.to_torch(in_place=False).to_dense()
    return mask * (left.to_torch(in_place=False) @ right.to_torch(in_place=False).T)


@pytest.fixture
def prebuilt_sddmm():
    import scorch_ops

    if getattr(scorch_ops, "sddmm_coo_float_prebuilt", None) is None:
        pytest.skip("this build has no prebuilt SDDMM kernel")
    return scorch_ops


@pytest.mark.usefixtures("fresh_kernel_caches")
def test_einsum_use_cache_false_skips_the_prebuilt_sddmm(monkeypatch, prebuilt_sddmm):
    monkeypatch.setattr(
        prebuilt_sddmm,
        "sddmm_coo_float_prebuilt",
        _refuse("the prebuilt SDDMM kernel"),
    )
    sampled, left, right = _sddmm_operands()
    result = scorch.einsum("ij,ik,jk->ij", sampled, left, right, use_cache=False)
    torch.testing.assert_close(
        result.to_torch(in_place=False).to_dense(),
        _sddmm_reference(sampled, left, right),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.parametrize("keywords", [{}, {"use_cache": True}], ids=["absent", "true"])
def test_einsum_use_cache_true_takes_the_prebuilt_sddmm(
    monkeypatch, prebuilt_sddmm, keywords
):
    original = prebuilt_sddmm.sddmm_coo_float_prebuilt
    calls = []

    def watched(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(prebuilt_sddmm, "sddmm_coo_float_prebuilt", watched)
    sampled, left, right = _sddmm_operands()
    result = scorch.einsum("ij,ik,jk->ij", sampled, left, right, **keywords)
    assert calls == [1]
    torch.testing.assert_close(
        result.to_torch(in_place=False).to_dense(),
        _sddmm_reference(sampled, left, right),
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.usefixtures("fresh_kernel_caches")
def test_matmul_use_cache_false_skips_the_prebuilt_kernels(monkeypatch):
    monkeypatch.setattr(
        ops, "resolve_prebuilt_matmul", _refuse("the prebuilt kernel resolution")
    )
    dense = _masked((24, 20), 0.15, 21)
    a = STensor.from_torch(dense, "A").to_sparse("ds")
    b = _masked((20, 6), 1.0, 22)
    result = scorch.matmul(a, b, use_cache=False)
    if isinstance(result, STensor):
        result = result.to_torch(in_place=False)
    torch.testing.assert_close(result, dense @ b, atol=1e-3, rtol=1e-3)
    # The same call without the keyword reaches the prebuilt kernels, so the call
    # above skipped a route that exists for these operands.
    with pytest.raises(AssertionError, match="prebuilt kernel resolution"):
        scorch.matmul(a, b)


@pytest.mark.parametrize("use_cache", [True, False])
def test_spmv_has_no_prebuilt_kernel_to_skip(monkeypatch, use_cache):
    for name in ("resolve_prebuilt_matmul", "execute_prebuilt_binary_kernel"):
        monkeypatch.setattr(ops, name, _refuse(name))
    dense = _masked((24, 20), 0.15, 31)
    vector = _masked((20,), 1.0, 32)
    a = STensor.from_torch(dense, "A").to_sparse("ds")
    x = STensor.from_torch(vector, "x")
    result = ops.spmv(a, x, use_cache=use_cache)
    torch.testing.assert_close(
        result.to_torch(in_place=False), dense @ vector, atol=1e-3, rtol=1e-3
    )
