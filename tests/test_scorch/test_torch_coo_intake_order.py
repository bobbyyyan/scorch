"""A torch COO tensor whose coalesced flag is set while its coordinates are out of order.

torch (2.13) keeps a COO tensor's coalesced flag through ``permute`` whenever the new
order leaves dimension 0 first, although the permute reorders the coordinates of a tensor
of rank 3 or more: ``coalesce()`` then does nothing, and the indices stay in the source's
order. ``STensor.from_torch`` permutes for a ``mode_order`` and then coalesces, so
``mode_order=[0, 2, 1]`` was refused with ``coo_coordinates_unordered``, and so was a COO
tensor a caller had permuted that way, through ``from_torch`` and ``from_coo``. A caller
can also set the flag on unordered indices directly (``is_coalesced=True``, which torch
does not check).

A torch COO tensor's intake now repairs such coordinates: sorted, with repeats summed, as
torch defines a COO tensor, into arrays of Scorch's own. The repair runs only when the
order screen that validates every COO tensor's arrays finds them out of order, so an
ordered tensor is built as before, sharing the caller's arrays. Raw COO arrays from a
caller are still refused when out of order.

The same flag was trusted by the structure a product under autograd builds for a 2-D
torch COO operand (autograd.py's ``_torch_operand``), where unsorted rows became wrong,
still increasing CSR positions and a wrong product with no error. That structure is now
built from the entries in canonical form when the same order screen finds disorder.

A flag a caller sets on unordered indices also misleads torch's own backward from the
tensor's ``values()`` to the leaf it was built from (torch 2.13), so the gradient of such
a tensor is checked at the values Scorch links, and its leaf against torch's own.

Every comparison is exact: the repair moves values and sums at most three of them, and
the products under autograd hold small integers.
"""

import itertools
import math

import pytest
import torch

import scorch
from scorch import stensor_grad
from scorch import storage as storage_module
from scorch.exceptions import RepeatedCoordinateError, TensorIndexError
from scorch.stensor import STensor
from scorch.storage import TensorIndex
from tests.test_scorch.level_array_support import logical_dense

F64 = torch.float64
SHAPES = {1: (9,), 2: (6, 7), 3: (5, 6, 7), 4: (4, 5, 3, 6)}
# More entries than the native screen's grain (65,536), so its thread-split pass runs.
LARGER = {1: (120000,), 2: (300, 400), 3: (48, 40, 64), 4: (12, 16, 20, 32)}
EXTENTS = [pytest.param(SHAPES, id="small"), pytest.param(LARGER, id="larger")]
FILL = 0.6
# Every order of rank 3 and 4. torch reorders the coordinates under a kept flag for the
# non-identity ones with 0 first; the others are controls that torch's coalesce sorts.
ORDERS = [
    pytest.param(order, id="".join(map(str, order)))
    for rank in (3, 4)
    for order in itertools.permutations(range(rank))
]
ENTRIES = ["from_torch", "from_coo"]


def distinct_entries(shape, seed, dtype=F64):
    """Distinct coordinates of ``shape`` in lexicographic order, ``[rank, count]`` int64,
    and a nonzero value for each."""

    generator = torch.Generator().manual_seed(seed)
    cells = math.prod(shape)
    count = max(3, int(cells * FILL))
    chosen, _ = torch.sort(torch.randperm(cells, generator=generator)[:count])
    indices = torch.stack(torch.unravel_index(chosen, shape))
    values = (torch.rand(count, generator=generator, dtype=F64) + 0.5).to(dtype)
    return indices, values


def coalesced_coo(shape, seed, dtype=F64):
    indices, values = distinct_entries(shape, seed, dtype)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def canonical(indices, values, shape):
    """torch's canonical form of these entries. A tensor built afresh carries no coalesced
    flag, so its ``coalesce()`` sorts the coordinates and sums the repeats."""

    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def convert(entry, tensor):
    if entry == "from_torch":
        return STensor.from_torch(tensor)
    return STensor.from_coo(coo_matrix=tensor)


def stored_levels(tensor):
    return torch.stack([level for (level,) in tensor.storage._mode_indices]).long()


def assert_stores(tensor, expected):
    """``tensor`` stores exactly ``expected``'s coordinates, strictly ascending in storage
    order, with ``expected``'s value at each."""

    assert torch.equal(stored_levels(tensor), expected.indices())
    assert torch.equal(tensor.values, expected.values())


def strictly_ascending(indices, shape):
    key = torch.zeros(indices.shape[1], dtype=torch.int64)
    for level, extent in zip(indices, shape):
        key = key * extent + level
    return bool((key[1:] > key[:-1]).all())


def flagged_out_of_order(shape, seed):
    """A COO tensor flagged coalesced whose distinct coordinates are shuffled, built as a
    caller can build one: ``is_coalesced=True`` is a promise torch does not check."""

    indices, values = distinct_entries(shape, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    shuffle = torch.randperm(indices.shape[1], generator=generator)
    assert not torch.equal(shuffle, torch.arange(shuffle.numel()))
    flagged = torch.sparse_coo_tensor(
        indices[:, shuffle], values[shuffle], shape, is_coalesced=True
    )
    assert flagged.is_coalesced()
    return flagged, indices, values


# -- every intake stores the coordinates in order ----------------------------------------


@pytest.mark.parametrize("dtype", [F64, torch.float32], ids=["float64", "float32"])
@pytest.mark.parametrize("extents", EXTENTS)
@pytest.mark.parametrize("order", ORDERS)
def test_from_torch_with_a_mode_order_stores_the_coordinates_in_order(
    order, extents, dtype
):
    shape = extents[len(order)]
    source = coalesced_coo(shape, seed=len(order), dtype=dtype)
    converted = STensor.from_torch(source, mode_order=list(order))
    assert [int(mode) for mode in converted.mode_order] == list(order)
    physical = tuple(shape[mode] for mode in order)
    assert_stores(
        converted, canonical(source.indices()[list(order)], source.values(), physical)
    )
    assert torch.equal(logical_dense(converted), source.to_dense().to(F64))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("extents", EXTENTS)
@pytest.mark.parametrize("order", ORDERS)
def test_a_coo_tensor_the_caller_permuted_converts_in_order(order, extents, entry):
    source = coalesced_coo(extents[len(order)], seed=10 + len(order))
    permuted = source.permute(*order)
    converted = convert(entry, permuted)
    assert_stores(
        converted,
        canonical(permuted._indices(), permuted._values(), tuple(permuted.shape)),
    )
    assert torch.equal(logical_dense(converted), source.to_dense().permute(*order))


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("extents", EXTENTS)
@pytest.mark.parametrize("rank", [1, 2, 3, 4])
def test_a_tensor_flagged_coalesced_out_of_order_converts_in_order(
    rank, extents, entry
):
    shape = extents[rank]
    flagged, indices, values = flagged_out_of_order(shape, seed=20 + rank)
    converted = convert(entry, flagged)
    expected = canonical(indices, values, shape)
    assert_stores(converted, expected)
    assert torch.equal(logical_dense(converted), expected.to_dense())


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("rank", [2, 3, 4])
def test_a_tensor_flagged_coalesced_with_a_repeat_is_summed_as_torch_sums_it(
    rank, entry
):
    shape = SHAPES[rank]
    indices, values = distinct_entries(shape, seed=30 + rank)
    # Entry 2 stores entry 0's coordinate again, after the larger one at entry 1.
    repeated = indices.clone()
    repeated[:, 2] = indices[:, 0]
    flagged = torch.sparse_coo_tensor(repeated, values, shape, is_coalesced=True)
    converted = convert(entry, flagged)
    expected = canonical(repeated, values, shape)
    assert expected._nnz() == values.numel() - 1
    assert_stores(converted, expected)


@pytest.mark.parametrize("route", ["mode_order", "permuted"])
def test_einsum_with_a_reordered_coo_operand_equals_torchs(route):
    source = coalesced_coo((6, 7, 5), seed=70)
    if route == "mode_order":
        operand = STensor.from_torch(source, mode_order=[0, 2, 1])
        dense = source.to_dense()
    else:
        # A torch COO operand, adopted by einsum itself.
        operand = source.permute(0, 2, 1)
        dense = source.to_dense().permute(0, 2, 1)
    x = torch.rand(
        dense.shape[2], generator=torch.Generator().manual_seed(71), dtype=F64
    )
    got = scorch.einsum("abc,c->ab", operand, STensor.from_torch(x), format="dd")
    torch.testing.assert_close(
        logical_dense(got), torch.einsum("abc,c->ab", dense, x), rtol=1e-12, atol=1e-12
    )


# -- the controls: ordered input is built as before; raw arrays are still refused -------


@pytest.mark.parametrize("screen", ["native", "absent"])
@pytest.mark.parametrize("route", ["mode_order", "permuted"])
@pytest.mark.parametrize("order", ORDERS)
def test_the_canonical_form_is_built_only_for_coordinates_out_of_order(
    monkeypatch, order, route, screen
):
    """Counted at ``_canonical_storage``, the one builder of the sorted copy. Whether the
    arrays torch hands over are out of order is read from them, not assumed, so the count
    stays right on a torch that clears the flag."""

    if screen == "native" and storage_module._NATIVE_LEX is None:
        pytest.skip("the extension has no lexicographic screen")
    if screen == "absent":
        monkeypatch.setattr(storage_module, "_NATIVE_LEX", None)
    built = []
    build = storage_module._canonical_storage

    def counted(*args):
        built.append(args)
        return build(*args)

    monkeypatch.setattr(storage_module, "_canonical_storage", counted)
    source = coalesced_coo(SHAPES[len(order)], seed=80)
    handed = source.permute(*order).coalesce()
    if route == "mode_order":
        converted = STensor.from_torch(source, mode_order=list(order))
    else:
        converted = STensor.from_torch(source.permute(*order))
    in_order = strictly_ascending(handed._indices(), tuple(handed.shape))
    assert len(built) == (0 if in_order else 1)
    assert_stores(
        converted, canonical(handed._indices(), handed._values(), tuple(handed.shape))
    )


def _from_components(shape, indices, values):
    return STensor.from_components(
        shape, "o" * len(shape), [[level] for level in indices], values
    )


def _hand_built_index(shape, indices, values):
    index = TensorIndex(
        tensor_format="o" * len(shape), mode_indices=[[level] for level in indices]
    )
    return STensor(shape=shape, index=index, value=values)


@pytest.mark.parametrize(
    "construct", [_from_components, _hand_built_index], ids=lambda f: f.__name__[1:]
)
def test_raw_coo_arrays_out_of_order_are_still_refused(construct):
    shape = SHAPES[3]
    indices, values = distinct_entries(shape, seed=40)
    swapped = indices.clone()
    swapped[:, [1, 2]] = indices[:, [2, 1]]
    with pytest.raises(TensorIndexError) as caught:
        construct(shape, swapped, values)
    assert caught.value.code == "coo_coordinates_unordered"
    assert not isinstance(caught.value, RepeatedCoordinateError)


def _lexicographic(levels, extent, distinct):
    key = torch.zeros(levels[0].numel(), dtype=torch.int64)
    for level in levels:
        key = key * extent + level.long()
    key = torch.unique(key) if distinct else torch.sort(key).values
    rebuilt = []
    for _ in levels:
        rebuilt.append((key % extent).to(levels[0].dtype))
        key = torch.div(key, extent, rounding_mode="floor")
    return rebuilt[::-1]


@pytest.mark.parametrize("screen", ["native", "absent"])
def test_the_repair_verdict_is_the_position_finders_verdict(monkeypatch, screen):
    """``_coo_order_violated`` against ``_first_coo_order_defect`` on random levels with
    three values each, so ties, repeats and descents are common, and on their sorted and
    sorted-distinct forms, which tie and strictly ascend."""

    if screen == "native" and storage_module._NATIVE_LEX is None:
        pytest.skip("the extension has no lexicographic screen")
    if screen == "absent":
        monkeypatch.setattr(storage_module, "_NATIVE_LEX", None)
    generator = torch.Generator().manual_seed(50)
    verdicts = []
    for rank in (1, 2, 3, 4):
        for count in (0, 1, 2, 3, 17, 200):
            for dtype in (torch.int32, torch.int64):
                for _ in range(4):
                    drawn = [
                        torch.randint(0, 3, (count,), generator=generator).to(dtype)
                        for _ in range(rank)
                    ]
                    cases = [drawn]
                    if count:
                        cases.append(_lexicographic(drawn, 3, distinct=False))
                        cases.append(_lexicographic(drawn, 3, distinct=True))
                    for levels in cases:
                        n = levels[0].numel()
                        violated = storage_module._coo_order_violated(levels, n)
                        defect = storage_module._first_coo_order_defect(levels, n)
                        assert violated == (defect is not None), (rank, n, levels)
                        verdicts.append(violated)
    assert any(verdicts) and not all(verdicts)


def test_coalesce_without_the_native_screen_returns_an_ordered_tensor_itself(
    monkeypatch,
):
    """``STensor.coalesce()`` asks the same repair verdict: with no screen to clear the
    arrays, the exact check still finds an ordered tensor ordered."""

    monkeypatch.setattr(storage_module, "_NATIVE_LEX", None)
    shape = SHAPES[3]
    indices, values = distinct_entries(shape, seed=60)
    tensor = _from_components(shape, indices, values)
    assert tensor.coalesce() is tensor
    for (level,) in tensor.storage._mode_indices:
        level[[0, -1]] = level[[-1, 0]]  # a descent, written in place
    stored = stored_levels(tensor)
    coalesced = tensor.coalesce()
    assert coalesced is not tensor
    assert_stores(coalesced, canonical(stored, tensor.values, shape))


# -- sharing and the gradient ------------------------------------------------------------


def test_a_repaired_tensor_shares_no_array_with_the_callers_and_an_ordered_one_does():
    source = coalesced_coo(SHAPES[3], seed=90)
    ordered = STensor.from_torch(source)
    for level, (stored,) in enumerate(ordered.storage._mode_indices):
        assert stored.data_ptr() == source._indices()[level].data_ptr()
    assert ordered._storage._value.data_ptr() == source._values().data_ptr()
    # torch's permute builds new indices, so a mode_order never shared the source's.
    permuted = source.permute(0, 2, 1)
    assert (
        permuted._indices().untyped_storage().data_ptr()
        != source._indices().untyped_storage().data_ptr()
    )
    repaired = STensor.from_torch(permuted)
    callers = {
        permuted._indices().untyped_storage().data_ptr(),
        permuted._values().untyped_storage().data_ptr(),
    }
    for (stored,) in repaired.storage._mode_indices:
        assert stored.untyped_storage().data_ptr() not in callers
    assert repaired._storage._value.untyped_storage().data_ptr() not in callers


@pytest.mark.parametrize("route", ["mode_order", "permuted"])
def test_the_gradient_reaches_the_callers_values_in_their_order(route):
    shape = (5, 6, 4)
    indices, values = distinct_entries(shape, seed=100)
    leaf = values.clone().requires_grad_()
    source = torch.sparse_coo_tensor(indices, leaf, shape).coalesce()
    want = torch.zeros(shape, dtype=F64)
    want[tuple(indices)] = values
    if route == "mode_order":
        got = STensor.from_torch(source, mode_order=[0, 2, 1]).to_torch(in_place=False)
    else:
        got = STensor.from_torch(source.permute(0, 2, 1)).to_torch(in_place=False)
        want = want.permute(0, 2, 1)
    upstream = torch.rand(
        want.shape, generator=torch.Generator().manual_seed(101), dtype=F64
    )
    (got * upstream).sum().backward()
    torch.testing.assert_close(got.detach(), want, rtol=0, atol=0)
    # Each given value's derivative is the upstream gradient at its own coordinate.
    at = upstream if route == "mode_order" else upstream.permute(0, 2, 1)
    torch.testing.assert_close(leaf.grad, at[tuple(indices)], rtol=0, atol=0)


FLAGGED = [
    # (0, 2, 1) at entries 0 and 2, and the order descends at entry 2.
    pytest.param([[0, 1, 0, 1], [2, 0, 2, 1], [1, 1, 1, 0]], id="repeat_twice"),
    # (0, 2, 1) at entries 0, 2 and 4.
    pytest.param(
        [[0, 1, 0, 1, 0], [2, 0, 2, 1, 2], [1, 1, 1, 0, 1]], id="repeat_three_times"
    ),
    pytest.param([[1, 0, 1], [0, 2, 1], [1, 1, 0]], id="out_of_order"),
]


@pytest.mark.parametrize("given", FLAGGED)
def test_the_values_a_flagged_tensor_links_get_each_coordinates_gradient(given):
    """The derivative of the values ``from_torch`` links (the flagged tensor's
    ``values()``) is the upstream gradient at each entry's coordinate, for every copy of a
    repeat.

    It is asserted there and not at the leaf the caller built the tensor from: torch's
    own backward from ``values()`` through ``sparse_coo_tensor(..., is_coalesced=True)``
    to that leaf trusts the flag, and is wrong for falsely flagged indices with or without
    a repeat (torch 2.13; probe_repeat_grad_v1.out). The leaf's gradient is asserted equal
    to the one torch's own ``to_dense()`` gives the same tensor: Scorch adds nothing to
    torch's error there, and follows a fix of it."""

    shape = (2, 3, 2)
    indices = torch.tensor(given)
    count = indices.shape[1]
    upstream = torch.rand(
        shape, generator=torch.Generator().manual_seed(110), dtype=F64
    )
    want = torch.zeros(shape, dtype=F64)
    want.index_put_(
        tuple(indices), torch.arange(1.0, count + 1, dtype=F64), accumulate=True
    )
    leaf = torch.arange(1.0, count + 1, dtype=F64).requires_grad_()
    flagged = torch.sparse_coo_tensor(indices, leaf, shape, is_coalesced=True)
    built = STensor.from_torch(flagged)
    moved = stensor_grad.history(built).grad_fn
    assert type(moved).__name__ == "_MovedValuesBackward"
    # The node of the values from_torch linked: _MovedValues' second input, its source.
    linked = moved.next_functions[1][0]
    seen = []
    linked.register_prehook(lambda grads: seen.append(grads[0].detach().clone()))
    got = built.to_torch(in_place=False)
    (got * upstream).sum().backward()
    torch.testing.assert_close(got.detach(), want, rtol=0, atol=0)
    (at_linked,) = seen
    torch.testing.assert_close(at_linked, upstream[tuple(indices)], rtol=0, atol=0)
    torch_leaf = torch.arange(1.0, count + 1, dtype=F64).requires_grad_()
    torch_flagged = torch.sparse_coo_tensor(
        indices, torch_leaf, shape, is_coalesced=True
    )
    (torch_flagged.to_dense() * upstream).sum().backward()
    torch.testing.assert_close(leaf.grad, torch_leaf.grad, rtol=0, atol=0)


# -- a 2-D torch COO operand of a product under autograd ---------------------------------

# Index arrays of three entries of a 4 x 3 matrix, flagged coalesced by the caller.
PRODUCT_CASES = {
    # torch's COO-to-CSR conversion reads rows [1, 0, 2] as positions [0, 2, 2, 3, 3]
    "rows_swapped": [[1, 0, 2], [0, 1, 2]],
    # and rows [3, 1, 0] as positions [0, 3, 3, 3, 3]
    "rows_descending": [[3, 1, 0], [2, 0, 1]],
    # (2, 1) stored twice, with the order descending at entry 1
    "repeat": [[2, 0, 2], [1, 2, 1]],
    "control_sorted": [[0, 1, 2], [1, 0, 2]],
}
PRODUCT_OTHER = {
    "a_at_matrix": [[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]],
    "a_at_vector": [1.0, 3.0, 7.0],
    "matrix_at_a": [[1.0, 2.0, 3.0, 4.0], [5.0, -6.0, 7.0, 8.0]],
}


def _product(matmul, form, a, other):
    result = matmul(other, a) if form == "matrix_at_a" else matmul(a, other)
    # A dense STensor result (scorch's matrix-vector product) as torch's, history kept.
    return result.to_torch(in_place=False) if isinstance(result, STensor) else result


@pytest.mark.parametrize("wanted", ["other", "values", "both"])
@pytest.mark.parametrize("form", list(PRODUCT_OTHER))
@pytest.mark.parametrize("case", list(PRODUCT_CASES))
def test_a_flagged_coo_operand_of_a_product_under_autograd_is_exact(case, form, wanted):
    """The product, the dense operand's gradient and the values' gradient, each exact
    against the same entries summed into a dense matrix, under an upstream gradient that
    differs at every position (a uniform one cannot see an entry placed in the wrong
    row)."""

    shape = (4, 3)
    indices = torch.tensor(PRODUCT_CASES[case])
    values = torch.tensor([1.0, 10.0, 100.0], dtype=F64)
    other0 = torch.tensor(PRODUCT_OTHER[form], dtype=F64)
    leaf = values.clone().requires_grad_(wanted != "other")
    other = other0.clone().requires_grad_(wanted != "values")
    a = torch.sparse_coo_tensor(indices, leaf, shape, is_coalesced=True)
    got = _product(scorch.matmul, form, a, other)

    leaf_ref = values.clone().requires_grad_()
    other_ref = other0.clone().requires_grad_()
    dense = torch.zeros(shape, dtype=F64).index_put(
        (indices[0], indices[1]), leaf_ref, accumulate=True
    )
    want = _product(torch.matmul, form, dense, other_ref)
    upstream = torch.arange(want.numel(), dtype=F64).reshape(want.shape) * 3.0 - 5.0
    (got * upstream).sum().backward()
    (want * upstream).sum().backward()
    torch.testing.assert_close(got.detach(), want.detach(), rtol=0, atol=0)
    if wanted != "values":
        torch.testing.assert_close(other.grad, other_ref.grad, rtol=0, atol=0)
    if wanted != "other":
        torch.testing.assert_close(leaf.grad, leaf_ref.grad, rtol=0, atol=0)
