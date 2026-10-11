"""``sparse_linear`` and ``sparse_linear_fm`` accept the same weights.

``sparse_linear_fm`` was documented ``weight: STensor`` and read
``weight.values.dtype`` directly, so a torch weight raised an unstructured
``AttributeError`` there, while ``sparse_linear`` -- which forwards to it --
converted a torch weight to a CSR STensor first.  The conversion now lives in
``sparse_linear_fm``, so both entry points take an STensor, a dense torch
tensor, or a torch CSR tensor, and a torch weight runs exactly the kernel the
equivalent STensor runs.
"""

import pytest
import torch

import scorch
from scorch.stensor import STensor

OUT_F, IN_F, BATCH = 20, 16, 6
ACTIVATIONS = (None, "relu", "sigmoid")


def _weight64(seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.rand(OUT_F, IN_F, generator=g, dtype=torch.float64)
    w[torch.rand(OUT_F, IN_F, generator=g) > 0.3] = 0
    w[3] = 0  # an empty output row
    return w


def _as(kind, w64, dtype):
    w = w64.to(dtype)
    if kind == "stensor":
        return STensor.from_csr(w.to_sparse_csr(), "W")
    if kind == "torch_dense":
        return w
    if kind == "torch_csr":
        return w.to_sparse_csr()
    raise AssertionError(kind)


def _reference(w64, x64, bias, activation, feature_major):
    y = w64 @ x64 if feature_major else x64 @ w64.t()
    if bias is not None:
        b = bias.double()
        y = y + (b[:, None] if feature_major else b)
    if activation == "relu":
        y = y.relu()
    elif activation == "sigmoid":
        y = y.sigmoid()
    return y


def _call(feature_major, x, w, bias, activation):
    if feature_major:
        return scorch.sparse_linear_fm(x, w, bias, activation)
    return scorch.sparse_linear(x, w, bias, activation)


@pytest.mark.parametrize("feature_major", [False, True], ids=["natural", "fm"])
@pytest.mark.parametrize("kind", ["stensor", "torch_dense", "torch_csr"])
@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_every_weight_kind_matches_float64(
    feature_major, kind, activation, with_bias, dtype
):
    w64 = _weight64(7)
    g = torch.Generator().manual_seed(11)
    shape = (IN_F, BATCH) if feature_major else (BATCH, IN_F)
    x = torch.rand(*shape, generator=g, dtype=torch.float64).to(dtype)
    bias = torch.rand(OUT_F, generator=g) if with_bias else None
    got = _call(feature_major, x, _as(kind, w64, dtype), bias, activation)
    expected = _reference(w64, x.double(), bias, activation, feature_major)
    assert got.shape == expected.shape
    torch.testing.assert_close(got.double(), expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("feature_major", [False, True], ids=["natural", "fm"])
@pytest.mark.parametrize("kind", ["torch_dense", "torch_csr"])
def test_a_torch_weight_runs_the_kernel_its_stensor_runs(feature_major, kind):
    # Bitwise equality: the torch weight is converted to the same CSR STensor,
    # so the same kernel sees the same arrays.
    w64 = _weight64(5)
    g = torch.Generator().manual_seed(3)
    shape = (IN_F, BATCH) if feature_major else (BATCH, IN_F)
    x = torch.rand(*shape, generator=g)
    bias = torch.rand(OUT_F, generator=g)
    by_stensor = _call(
        feature_major, x, _as("stensor", w64, torch.float32), bias, "relu"
    )
    by_torch = _call(feature_major, x, _as(kind, w64, torch.float32), bias, "relu")
    assert torch.equal(by_torch, by_stensor)
