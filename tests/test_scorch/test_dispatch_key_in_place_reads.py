"""The einsum dispatch key reads an STensor's layout, dtype and device in place (series
step 31c), and tells operands apart exactly as it did.

Every einsum that reaches a kernel builds ``ops._einsum_cache_key``.  It used to read
each operand's ``layout``, ``dtype`` and ``device`` properties through ``getattr``, in
two generators and a nested function, and to key the dtype and device by their strings.
It now reads an exact STensor's three values from its metadata, where the properties
read them, and keys torch's own dtypes and devices as objects.  A dtype or device given
as a string by a tensor-like still keys as that string, so it no longer keys equal to
torch's object of the same name: the key tells more apart, never less.  These tests pin:

* an STensor's key is the key of a tensor-like that carries the same layout, dtype and
  device and goes through the general path;
* two of torch's dtypes, or two of its devices, key equal exactly when their strings are
  equal, and equal keys hash equal; any other value keys as its string;
* STensors that differ only in dtype, or only in layout, key differently.
"""

from __future__ import annotations

from typing import Any

import torch

import scorch.ops as ops  # type: ignore[import-untyped]
from scorch import STensor  # type: ignore[import-untyped]


def _dense(dtype: torch.dtype = torch.float64) -> STensor:
    generator = torch.Generator().manual_seed(3)
    return STensor.from_torch(torch.rand(4, 5, generator=generator).to(dtype))


class _Carrier:
    """A tensor-like carrying an STensor's layout, dtype and device as attributes."""

    def __init__(self, tensor: STensor) -> None:
        self.layout = tensor.layout
        self.dtype = tensor.dtype
        self.device = tensor.device


def _key(*tensors: Any) -> tuple:
    return ops._einsum_cache_key("ij,jk->ik", tensors, "dd", None, None)


def test_an_stensors_key_is_the_general_paths_key() -> None:
    a = _dense()
    b = _dense().to_sparse("ds")
    assert _key(a, b) == _key(_Carrier(a), _Carrier(b))
    assert hash(_key(a, b)) == hash(_key(_Carrier(a), _Carrier(b)))


def test_torch_dtypes_and_devices_key_equal_exactly_when_their_strings_are() -> None:
    groups = [
        [torch.float32, torch.float, torch.float64, torch.double, torch.int32],
        [torch.device("cpu"), torch.device("cpu"), torch.device("meta")],
    ]
    for group in groups:
        for first in group:
            for second in group:
                same = ops._runtime_key(first) == ops._runtime_key(second)
                assert same is (str(first) == str(second))
                if same:
                    assert hash(ops._runtime_key(first)) == hash(
                        ops._runtime_key(second)
                    )
    for other in (None, "cpu", "torch.float32", 3):
        assert ops._runtime_key(other) == str(other)


def test_stensors_differing_in_dtype_or_layout_key_differently() -> None:
    a64, a32 = _dense(torch.float64), _dense(torch.float32)
    assert _key(a64, a64) != _key(a32, a32)
    assert _key(a64, a64) != _key(a64.to_sparse("ds"), a64)
