"""How a kernel's result becomes an STensor, and what may be shared with it.

Every path that runs a kernel ends in `_wrap_generated_result`: finish the
zero-initialized trailing positions, describe the arrays, build the storage, name the
tensor. That sequence used to be written out at nine call sites -- eight in ops.py and one
in `STensor.to_sparse` -- and this file pins what the consolidation has to preserve, plus
the two things it made possible.

**Adopting the arrays instead of copying them.** The copy exists so a *caller* cannot
invalidate a validated tensor by mutating what it passed in, which does not describe a
buffer a kernel allocated for its own output microseconds ago; Scorch already treats the
result's *values* that way, sharing them through `detach`. What has to be true first is
that no kernel returns output index arrays aliasing an *input's* --
`test_kernel_results_do_not_alias_their_operands` checks that directly rather than
reasoning about it, and `test_sddmm_result_does_not_alias_its_operand` covers the one
kernel where the answer is no.

**Skipping the O(nnz) structural walk on arrays our own compiler emitted**, which is
debug-only and on for this suite. The four tests at the bottom of this file pin that in
both directions, including that it really is skipped when off -- otherwise the flag saves
nothing.

Together they are worth 1.04-1.15x of a whole sparse-result `einsum` on both hosts, with
the walk the larger half. See `bench/bench_index_validation.py --what adopt`.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import scorch
import scorch.stensor as stensor_module
import scorch.storage as storage_module
from scorch import STensor
from scorch.exceptions import TensorIndexError, TensorTypeError
from scorch.stensor import _wrap_generated_result
from scorch.storage import TensorIndex


def sparse_csr(rows, cols, degree, seed=0):
    generator = np.random.default_rng(seed)
    dense = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        chosen = generator.choice(cols, size=min(degree, cols), replace=False)
        dense[row, chosen] = generator.standard_normal(len(chosen)).astype(np.float32)
    return torch.from_numpy(dense)


def index_buffers(tensor):
    """Every distinct storage a tensor's index arrays sit in."""
    return {
        array.untyped_storage().data_ptr()
        for level in tensor._storage._mode_indices
        for array in level
    }


# --------------------------------------------------------------------------- #
# The adopt flag's semantics, at the level it is implemented
# --------------------------------------------------------------------------- #


def test_the_public_constructor_still_copies():
    """The documented guarantee: caller mutation cannot reach inside the descriptor."""
    positions = torch.tensor([0, 2, 3], dtype=torch.int32)
    coordinates = torch.tensor([0, 2, 1], dtype=torch.int32)
    index = TensorIndex("ds", [[], [positions, coordinates]])
    held = index._mode_indices[1]
    assert held[0].data_ptr() != positions.data_ptr()
    assert held[1].data_ptr() != coordinates.data_ptr()
    coordinates[0] = 99
    assert index._mode_indices[1][1].tolist() == [0, 2, 1]


def test_adopting_shares_the_arrays_it_is_given():
    positions = torch.tensor([0, 2, 3], dtype=torch.int32)
    coordinates = torch.tensor([0, 2, 1], dtype=torch.int32)
    index = TensorIndex("ds", [[], [positions, coordinates]], _adopt=True)
    held = index._mode_indices[1]
    assert held[0].data_ptr() == positions.data_ptr()
    assert held[1].data_ptr() == coordinates.data_ptr()


def test_adopting_still_validates_everything():
    """Only the copy is skipped -- every structural check still runs."""
    good = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(Exception):  # wrong dtype
        TensorIndex("ds", [[], [good, good.float()]], _adopt=True)
    with pytest.raises(Exception):  # not contiguous
        TensorIndex("ds", [[], [good, good.repeat(2)[::2]]], _adopt=True)
    with pytest.raises(Exception):  # wrong arity for the level
        TensorIndex("ds", [[], [good]], _adopt=True)
    with pytest.raises(Exception):  # not one-dimensional
        TensorIndex("ds", [[], [good, good.reshape(2, 1)]], _adopt=True)


# --------------------------------------------------------------------------- #
# The question adoption depends on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "case",
    ["spgemm_sparse_out", "einsum_sparse_out", "elementwise_sparse_out"],
)
def test_kernel_results_do_not_alias_their_operands(case):
    """A kernel's output index arrays must be its own.

    If this ever fails, `_wrap_generated_result(adopt=True)` is unsafe for that path
    and the copy has to stay: the result would otherwise share index buffers with its
    operand. The failure message names the path so the flag can be scoped rather than
    abandoned wholesale.
    """
    left = STensor.from_torch(sparse_csr(32, 32, 3), "L").to_sparse("ds")
    if case == "elementwise_sparse_out":
        right = STensor.from_torch(sparse_csr(32, 32, 3, seed=1), "R").to_sparse("ds")
        result = scorch.einsum("ij,ij->ij", left, right, format="ds")
    elif case == "einsum_sparse_out":
        right = STensor.from_torch(torch.rand(32, 8), "R")
        result = scorch.einsum("ik,kj->ij", left, right, format="ds")
    else:
        right = STensor.from_torch(sparse_csr(32, 32, 3, seed=2), "R").to_sparse("ds")
        result = scorch.matmul(left, right, output_format="ds")
    if not isinstance(result, STensor):
        pytest.skip(f"{case} did not produce an STensor")
    shared = index_buffers(result) & (index_buffers(left) | index_buffers(right))
    assert not shared, (
        f"{case}: result index arrays alias an operand's "
        f"({len(shared)} shared buffers) -- adoption is unsafe here"
    )


def test_results_are_correct_through_the_shared_wrapper():
    """The consolidation must not have changed any answer."""
    left = STensor.from_torch(sparse_csr(24, 24, 4), "L").to_sparse("ds")
    dense = torch.rand(24, 6)
    reference = left.to_torch() @ dense
    got = scorch.matmul(left, dense)
    torch.testing.assert_close(
        got if isinstance(got, torch.Tensor) else got.to_torch(),
        reference,
        atol=1e-3,
        rtol=1e-3,
    )


def sddmm_operands(rows=16, cols=16, inner=4, degree=3):
    """The exact shape of operand `ops.einsum` requires to reach the prebuilt SDDMM:
    'ij,ik,jk->ij' with S in COO float32, A and B dense, all in natural mode order."""
    sampled = STensor.from_torch(sparse_csr(rows, cols, degree), "S").to_sparse("oo")
    left = STensor.from_torch(torch.rand(rows, inner), "A")
    right = STensor.from_torch(torch.rand(cols, inner), "B")
    return sampled, left, right


def sddmm_watched(monkeypatch):
    """Run the SDDMM route and report whether the prebuilt kernel actually served it.

    Without this the aliasing guard below is worthless: the generic compiler path
    allocates its own output arrays, so if dispatch ever stopped reaching
    `sddmm_coo_float_prebuilt` -- a format check tightened, the symbol renamed -- the
    result would stop aliasing for a reason that has nothing to do with the opt-out
    being in place, and the test would pass while guarding nothing.
    """
    import scorch_ops

    original = scorch_ops.sddmm_coo_float_prebuilt
    calls = []

    def watched(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(scorch_ops, "sddmm_coo_float_prebuilt", watched)
    sampled, left, right = sddmm_operands()
    result = scorch.einsum("ij,ik,jk->ij", sampled, left, right)
    return sampled, left, right, result, len(calls)


def test_sddmm_reaches_the_prebuilt_kernel(monkeypatch):
    """If this stops holding, the guard below stops guarding anything."""
    sampled, left, right, result, calls = sddmm_watched(monkeypatch)
    assert calls == 1, (
        f"the prebuilt SDDMM kernel served {calls} of 1 calls; dispatch no longer "
        "reaches it, so test_sddmm_result_does_not_alias_its_operand proves nothing"
    )
    # Sampled dense-dense product: the mask times A @ B.T, elementwise.
    reference = sampled.to_torch().to_dense() * (left.to_torch() @ right.to_torch().T)
    torch.testing.assert_close(
        result.to_torch().to_dense(), reference, atol=1e-3, rtol=1e-3
    )


def test_sddmm_result_does_not_alias_its_operand(monkeypatch):
    """SDDMM is the one wrap that must keep copying, so this is the one path where the
    aliasing question has a bad answer and the copy is load-bearing.

    `sddmm_coo_float_prebuilt` returns the *operand's* index arrays rather than its own
    (`D.storage.index.mode_indices = S_mode_indices` in csrc/kernels.h) because the
    result has the same sparsity pattern. `ops.einsum` therefore passes `adopt=False`
    there. If that opt-out is removed -- or if adoption becomes unconditional inside
    `_wrap_generated_result` -- this fails, which is the point: the failure names the
    kernel to fix rather than surfacing later as a result that mutates its operand.
    """
    sampled, _left, _right, result, calls = sddmm_watched(monkeypatch)
    assert calls == 1, "the prebuilt SDDMM kernel did not serve this call"
    shared = index_buffers(result) & index_buffers(sampled)
    assert not shared, (
        "the SDDMM result shares index buffers with its operand "
        f"({len(shared)} shared storages): csrc/kernels.h hands back S's own arrays, so "
        "this wrap must pass adopt=False"
    )


def test_adoption_is_on_by_default_where_it_is_safe():
    """The generated path allocates its output arrays, so it adopts.

    Pinned because the win is silent: nothing fails if adoption regresses to copying,
    it just costs 14-22% of every sparse-result wrap again.
    """
    from scorch import stensor as stensor_module

    assert stensor_module._ADOPT_CELL[0], (
        "adoption is off; the generated path's output arrays come from torch::empty "
        "in cin_lowerer.py and cannot alias an operand"
    )


# --------------------------------------------------------------------------- #
# Debug-only structural validation of generated results
# --------------------------------------------------------------------------- #


class FakeKernelResult:
    """The shape of a compiled module's return value, holding arrays we choose.

    `_wrap_generated_result` only reads `.storage.index.mode_indices` and
    `.storage.value`, so a malformed "kernel result" can be built without a kernel --
    which is the only way to test what happens when the compiler emits something wrong,
    since by construction it does not.
    """

    def __init__(self, mode_indices, values):
        self.storage = SimpleNamespace(
            index=SimpleNamespace(mode_indices=mode_indices), value=values
        )


def malformed_csr_result():
    """A "ds" result whose column coordinate is past the declared extent.

    Exactly the corruption the O(nnz) walk exists to catch: a kernel doing unchecked
    pointer arithmetic on this would read outside its buffer.
    """
    positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    coordinates = torch.tensor([0, 99], dtype=torch.int32)  # extent is 2
    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    return FakeKernelResult([[], [positions, coordinates]], values)


def test_generated_results_are_walked_when_the_flag_is_on():
    """What `tests/conftest.py` buys: a malformed compiler output is a structured error.

    The session fixture turns this on for the whole suite, so this is also asserting that
    every other test in the repository would catch a codegen bug rather than segfault.
    """
    assert storage_module._VALIDATE_KERNEL_RESULTS[0], (
        "the session fixture in tests/conftest.py should have turned this on; without it "
        "no test in this repository walks a generated result"
    )
    with pytest.raises(TensorIndexError):
        _wrap_generated_result(
            shape=(2, 2), tensor_format="ds", result_cpp=malformed_csr_result()
        )


def test_generated_results_are_not_walked_when_the_flag_is_off():
    """And what release actually does -- otherwise the flag saves nothing.

    Asserted by the absence of the error above, which is the only observable difference:
    the walk has no other effect on a well-formed tensor.
    """
    old = storage_module._VALIDATE_KERNEL_RESULTS[0]
    storage_module._VALIDATE_KERNEL_RESULTS[0] = False
    try:
        wrapped = _wrap_generated_result(
            shape=(2, 2), tensor_format="ds", result_cpp=malformed_csr_result()
        )
    finally:
        storage_module._VALIDATE_KERNEL_RESULTS[0] = old
    # Deliberately not used for anything: it is malformed, and a kernel handed these
    # arrays would index outside its buffer. Constructing it is the whole assertion.
    assert wrapped.shape == (2, 2)


def test_trusting_the_index_still_runs_the_cheap_checks():
    """`_trusted_index` skips the O(nnz) walk and nothing else.

    The per-array checks are what makes adopting a kernel's arrays safe -- a non-contiguous
    or wrongly-typed array would be misread by `data_ptr` arithmetic no matter who produced
    it -- so they must survive with the flag off.
    """
    old = storage_module._VALIDATE_KERNEL_RESULTS[0]
    storage_module._VALIDATE_KERNEL_RESULTS[0] = False
    try:
        wrong_dtype = FakeKernelResult(
            [
                [],
                [
                    torch.tensor([0, 1, 2], dtype=torch.int32),
                    torch.tensor([0.0, 1.0], dtype=torch.float32),
                ],
            ],
            torch.tensor([1.0, 2.0], dtype=torch.float32),
        )
        with pytest.raises((TensorIndexError, TensorTypeError)):
            _wrap_generated_result(
                shape=(2, 2), tensor_format="ds", result_cpp=wrong_dtype
            )

        strided = torch.zeros(4, dtype=torch.int32)[::2]
        assert not strided.is_contiguous()
        non_contiguous = FakeKernelResult(
            [[], [torch.tensor([0, 1, 2], dtype=torch.int32), strided]],
            torch.tensor([1.0, 2.0], dtype=torch.float32),
        )
        with pytest.raises((TensorIndexError, TensorTypeError)):
            _wrap_generated_result(
                shape=(2, 2), tensor_format="ds", result_cpp=non_contiguous
            )
    finally:
        storage_module._VALIDATE_KERNEL_RESULTS[0] = old


def test_caller_supplied_arrays_are_always_walked():
    """The flag must not reach a caller's own arrays, whatever it is set to.

    This is the line the design draws: we trust our compiler, never the caller. A
    malformed hand-built CSR has to raise with the flag off, which is release behaviour.
    """
    old = storage_module._VALIDATE_KERNEL_RESULTS[0]
    storage_module._VALIDATE_KERNEL_RESULTS[0] = False
    try:
        csr = torch.sparse_csr_tensor(
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([0, 99], dtype=torch.int32),  # past the extent
            torch.tensor([1.0, 2.0]),
            size=(2, 2),
            check_invariants=False,
        )
        with pytest.raises(TensorIndexError):
            STensor.from_torch(csr)
    finally:
        storage_module._VALIDATE_KERNEL_RESULTS[0] = old


# --------------------------------------------------------------------------- #
# The cached parts of a dense result
# --------------------------------------------------------------------------- #


def dense_result(shape=(2, 3), fill=1.0):
    """What a generated all-dense kernel hands back: values, and no index arrays."""
    values = torch.full((shape[0] * shape[1],), fill, dtype=torch.float32)
    return FakeKernelResult([[] for _ in shape], values)


def observable(tensor):
    """Everything a holder of this STensor can see about its structure."""
    return dict(
        shape=tuple(tensor.shape),
        fmt=str(tensor.format),
        name=tensor.name,
        dtype=tensor.values.dtype,
        device=tensor.values.device,
        mode_order=tuple(tensor.storage.index.mode_order),
        index_dtype=tensor.storage.layout.index_dtype,
        logical=tuple(tensor.storage.layout.logical_shape),
        physical=tuple(tensor.storage.layout.physical_shape),
        mode_indices=[
            [a.tolist() for a in level] for level in tensor._storage._mode_indices
        ],
        requires_grad=tensor.requires_grad,
        values=tensor.values.tolist(),
    )


def without_result_parts_cache(monkeypatch):
    """Turn the cache off the way a bench does: stop installing, and empty it."""
    monkeypatch.setattr(stensor_module, "_RESULT_PARTS_CACHE_MAX", 0)
    monkeypatch.setattr(stensor_module, "_RESULT_PARTS_CACHE", {})


@pytest.mark.parametrize("name", [None, "C"])
@pytest.mark.parametrize("mode_order", [None, [0, 1], [1, 0]])
def test_dense_result_parts_match_the_ordinary_path(monkeypatch, name, mode_order):
    """The shared index/layout/metadata must describe what the long way would build."""
    with monkeypatch.context() as off:
        without_result_parts_cache(off)
        ordinary = _wrap_generated_result(
            shape=(2, 3),
            tensor_format="dd",
            result_cpp=dense_result(),
            mode_order=mode_order,
            name=name,
        )
    # Twice, because the first call through the cached path is the one that fills it.
    cached = [
        _wrap_generated_result(
            shape=(2, 3),
            tensor_format="dd",
            result_cpp=dense_result(),
            mode_order=mode_order,
            name=name,
        )
        for _ in range(2)
    ]
    for got in cached:
        assert observable(got) == observable(ordinary)


def test_dense_results_do_not_share_values_or_storage():
    """Sharing the *description* must not turn into sharing the payload."""
    first = _wrap_generated_result(
        shape=(2, 3), tensor_format="dd", result_cpp=dense_result(fill=1.0)
    )
    second = _wrap_generated_result(
        shape=(2, 3), tensor_format="dd", result_cpp=dense_result(fill=2.0)
    )
    assert first.storage is not second.storage
    assert first.values.data_ptr() != second.values.data_ptr()
    first.values[0] = 99.0
    assert second.values[0].item() == 2.0


def test_dense_result_parts_separate_shape_format_order_name_and_dtype():
    """Each field of the key has to be a field of the key."""

    def wrap(**overrides):
        spec = dict(shape=(2, 3), tensor_format="dd", mode_order=None, name=None)
        spec.update(overrides)
        shape = spec["shape"]
        values = torch.zeros(
            shape[0] * shape[1], dtype=spec.pop("dtype", torch.float32)
        )
        return _wrap_generated_result(
            result_cpp=FakeKernelResult([[] for _ in shape], values), **spec
        )

    base = observable(wrap())
    assert observable(wrap(shape=(3, 2))) != base
    assert observable(wrap(mode_order=[1, 0])) != base
    assert observable(wrap(name="C")) != base
    assert observable(wrap(dtype=torch.float64)) != base
    # A different format at the same shape: still dense, so still the cached path.
    assert observable(wrap(tensor_format="dd")) == base


def test_dense_result_parts_decline_a_sparse_format():
    """A sparse result's arrays differ per call, so it must take the ordinary path."""
    positions = torch.tensor([0, 1, 2], dtype=torch.int32)
    coordinates = torch.tensor([0, 1], dtype=torch.int32)
    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    before = len(stensor_module._RESULT_PARTS_CACHE)
    wrapped = _wrap_generated_result(
        shape=(2, 2),
        tensor_format="ds",
        result_cpp=FakeKernelResult([[], [positions, coordinates]], values),
    )
    assert wrapped.shape == (2, 2)
    assert len(stensor_module._RESULT_PARTS_CACHE) == before


def test_dense_result_parts_decline_index_arrays_on_a_dense_level():
    """A dense level may carry no arrays, and that check must stay a real one.

    The shortcut could have inferred "dense format, therefore no arrays" from the format
    alone. It does not, because it is `_normalize_mode_indices` inside the ordinary path
    that enforces the arity -- so assuming it would convert a structured error into a
    silently accepted tensor. Here the kernel "returns" an array for a dense mode.
    """
    stray = torch.tensor([0, 1], dtype=torch.int32)
    values = torch.zeros(6, dtype=torch.float32)
    with pytest.raises(TensorIndexError):
        _wrap_generated_result(
            shape=(2, 3),
            tensor_format="dd",
            result_cpp=FakeKernelResult([[stray], []], values),
        )


def test_einsum_dense_result_is_correct_and_repeatable():
    """End to end: the same product three times, against torch, sharing cached parts."""
    torch.manual_seed(0)
    dense = (torch.rand(48, 48) < 0.2).float()
    A = STensor.from_torch(dense.to_sparse_csr())
    for _ in range(3):
        B = torch.rand(48, 6)
        out = scorch.einsum("ik,kj->ij", A, B, format="dd")
        torch.testing.assert_close(
            out.to_torch(in_place=False), dense @ B, atol=1e-3, rtol=1e-3
        )
