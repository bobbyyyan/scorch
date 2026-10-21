"""A kernel built by ``einsum(..., compile_only=True)`` on TensorSpecs is the kernel the
same call runs on STensors of that layout, dtype and device, with no second build.

Series step 31c changed how ``ops._einsum_cache_key`` reads an operand (an STensor's
layout, dtype and device from its metadata; torch's dtypes and devices as objects).
Precompiling on specs and then running on tensors is the workflow a change to that key
could break without any result changing, so these tests count builds.  They hold on the
tree before step 31c as well.  They pin:

* after a compile-only call on specs, the same call on STensors whose layout, dtype and
  device are the specs' builds nothing, and a second such call builds nothing and reads
  the one dispatch entry the first made; the results equal the float64 product;
* specs and STensors of one layout, dtype and device give one dispatch key.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest
import torch

import scorch
import scorch.ops as ops  # type: ignore[import-untyped]
from scorch import STensor  # type: ignore[import-untyped]
from scorch.layout import TensorSpec  # type: ignore[import-untyped]


def _sparse(rows: int, cols: int) -> STensor:
    generator = torch.Generator().manual_seed(11)
    dense = torch.rand(rows, cols, generator=generator, dtype=torch.float64)
    dense[dense < 0.5] = 0.0
    return STensor.from_torch(dense).to_sparse("ds")


def _dense(*shape: int) -> STensor:
    generator = torch.Generator().manual_seed(12)
    values = torch.rand(*shape, generator=generator, dtype=torch.float64)
    return STensor.from_torch(values)


def _spec_of(tensor: STensor, name: str) -> TensorSpec:
    """A spec with the tensor's format, shape, mode order, both dtypes and device."""
    return TensorSpec(
        tensor.format,
        tuple(tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
        mode_order=tuple(tensor.mode_order),
        index_dtype=tensor.index_dtype,
        name=name,
    )


# Each case: expression, output format, operands.
_CASES = {
    "spmm": ("ik,kj->ij", "dd", lambda: (_sparse(6, 5), _dense(5, 4))),
    "spmv": ("ij,j->i", "d", lambda: (_sparse(6, 5), _dense(5))),
}


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    """Every kernel einsum loads, counted; the compiler's caches start empty."""
    loaded: List[Any] = []
    real = ops._load_validated_prepared_kernel

    def counted(prepared: Any) -> Any:
        loaded.append(prepared)
        return real(prepared)

    monkeypatch.setattr(ops, "_load_validated_prepared_kernel", counted)
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
    return loaded


@pytest.mark.parametrize("case", sorted(_CASES))
def test_stensors_run_the_kernel_precompiled_on_their_specs(
    case: str, builds: List[Any]
) -> None:
    expression, output_format, make = _CASES[case]
    tensors: Tuple[STensor, ...] = make()
    specs = [_spec_of(t, f"t{index}") for index, t in enumerate(tensors)]
    spec_result = scorch.einsum(
        expression, *specs, compile_only=True, format=output_format
    )
    assert isinstance(spec_result, TensorSpec)
    assert len(builds) == 1

    # ``to_torch`` densifies the operand's own storage unless asked not to, after which
    # the calls below would be all-dense and torch's (``_dense_einsum``), not the
    # precompiled kernel's.
    want = torch.einsum(expression, *(t.to_torch(in_place=False) for t in tensors))
    assert [t.format for t in tensors] == [spec.format for spec in specs]
    for call in range(2):
        got = scorch.einsum(expression, *tensors, format=output_format)
        assert len(builds) == 1, f"call {call} built a kernel"
        assert len(ops._einsum_dispatch_cache) == 1
        assert torch.allclose(got.to_torch(), want, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_specs_and_stensors_of_one_layout_give_one_dispatch_key(case: str) -> None:
    expression, output_format, make = _CASES[case]
    tensors = make()
    specs = tuple(_spec_of(t, f"t{index}") for index, t in enumerate(tensors))
    by_specs = ops._einsum_cache_key(expression, specs, output_format, None, None)
    by_tensors = ops._einsum_cache_key(expression, tensors, output_format, None, None)
    assert by_specs == by_tensors
    assert hash(by_specs) == hash(by_tensors)
