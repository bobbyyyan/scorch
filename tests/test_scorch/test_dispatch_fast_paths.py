"""The per-call fast paths must be indistinguishable from the paths they replace.

Three caches shortcut work that a small ``scorch.matmul`` was repeating on every call:
``parse_format`` hands back a shared ``TensorFormat``, ``TensorLayout.from_physical_shape``
hands back a shared layout, and ``STensor.from_torch`` assembles a dense operand from
cached immutable parts instead of re-deriving them. Each is only legitimate if what comes
out is what the ordinary path would have produced, and if a shared value object cannot be
mutated through one holder and observed through another.
"""

import torch

import scorch
from scorch import STensor, TensorFormat, TensorLayout, parse_format
from scorch.stensor import _DENSE_PARTS_CACHE


def _fields(t):
    """Everything observable about an STensor's structure, for a field-by-field diff."""
    storage = t.storage
    return dict(
        shape=tuple(t.shape),
        fmt=str(t.format),
        dtype=t.values.dtype,
        device=t.values.device,
        mode_order=tuple(storage.index.mode_order),
        index_dtype=storage.layout.index_dtype,
        logical=tuple(storage.layout.logical_shape),
        physical=tuple(storage.layout.physical_shape),
        n_levels=len(storage.index.mode_indices),
        name=t.name,
    )


def test_dense_from_torch_cached_matches_first_build():
    """The cached assembly agrees with the build that populated the cache."""
    _DENSE_PARTS_CACHE.clear()
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cold = STensor.from_torch(x, "A")  # populates
    warm = STensor.from_torch(x, "A")  # assembles from the cache
    assert _fields(cold) == _fields(warm)
    assert torch.equal(cold.values, warm.values)
    assert torch.equal(cold.to_torch(in_place=False), warm.to_torch(in_place=False))


def test_dense_from_torch_cache_key_separates_shape_dtype_name_and_order():
    """Anything the cached parts depend on has to be part of the key."""
    _DENSE_PARTS_CACHE.clear()
    base = torch.zeros((2, 3), dtype=torch.float32)
    variants = [
        STensor.from_torch(base, "A"),
        STensor.from_torch(torch.zeros((3, 2), dtype=torch.float32), "A"),
        STensor.from_torch(torch.zeros((2, 3), dtype=torch.float64), "A"),
        STensor.from_torch(base, "B"),
        STensor.from_torch(base, "A", mode_order=[1, 0]),
    ]
    seen = {tuple(sorted(_fields(v).items(), key=lambda kv: kv[0])) for v in variants}
    assert len(seen) == len(variants), "a key component is not distinguishing entries"


def test_dense_from_torch_values_are_not_shared_between_tensors():
    """Only the immutable parts are shared; two operands must not alias each other."""
    _DENSE_PARTS_CACHE.clear()
    first = STensor.from_torch(torch.ones((2, 2), dtype=torch.float32))
    second = STensor.from_torch(torch.full((2, 2), 5.0, dtype=torch.float32))
    assert first.values.data_ptr() != second.values.data_ptr()
    assert torch.equal(first.values, torch.ones(4))
    assert torch.equal(second.values, torch.full((4,), 5.0))


def test_dense_from_torch_reflects_later_writes_like_the_ordinary_path():
    """Values alias the source tensor exactly as the uncached construction did."""
    _DENSE_PARTS_CACHE.clear()
    x = torch.zeros((2, 2), dtype=torch.float32)
    STensor.from_torch(x)  # populate, so the next call takes the cached branch
    y = torch.zeros((2, 2), dtype=torch.float32)
    wrapped = STensor.from_torch(y)
    y[0, 0] = 7.0
    assert wrapped.values[0].item() == 7.0


def test_parse_format_returns_equal_formats_for_every_spelling():
    assert parse_format("ds") == parse_format("d,s") == TensorFormat("ds")
    assert parse_format(["dense", "compressed"]) == parse_format("ds")
    already = TensorFormat("oo")
    assert parse_format(already) is already


def test_layout_cache_returns_equal_layouts_and_rejects_bad_input():
    a = TensorLayout.from_physical_shape((4, 5), "ds")
    b = TensorLayout.from_physical_shape([4, 5], "ds")
    assert a == b
    assert a.logical_shape == (4, 5)
    # A permutation is part of the identity, not something the cache may ignore.
    p = TensorLayout.from_physical_shape((4, 5), "ds", permutation=[1, 0])
    assert p.logical_shape == (5, 4)
    assert p != a


def test_shared_layout_cannot_be_mutated_through_a_holder():
    """The cached parts are frozen; an attempt to write one must still raise."""
    layout = TensorLayout.from_physical_shape((2, 2), "dd")
    for field, value in (("index_dtype", torch.int64), ("permutation", (1, 0))):
        try:
            setattr(layout, field, value)
        except Exception:
            continue
        raise AssertionError(f"{field} was mutable on a shared TensorLayout")


def test_matmul_result_unchanged_across_repeated_calls():
    """The caches must not make the second call of a repeated matmul differ."""
    torch.manual_seed(0)
    dense = (torch.rand(24, 24) < 0.3).float()
    A = STensor.from_torch(dense.to_sparse_csr())
    B = torch.rand(24, 6)
    reference = torch.matmul(dense.double(), B.double()).float()
    outs = [scorch.matmul(A, B) for _ in range(3)]
    for out in outs:
        got = out if isinstance(out, torch.Tensor) else out.to_torch(in_place=False)
        torch.testing.assert_close(got, reference, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(got, outs[0], atol=0, rtol=0)


def test_raw_values_matches_values():
    """_raw_values skips a defensive copy, not a transformation."""
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    t = STensor.from_torch(x)
    assert torch.equal(t._raw_values, t.values)
    assert t._raw_values.dtype == t.values.dtype
    assert not t._raw_values.requires_grad
