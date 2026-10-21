"""A sparse result's levels bounded before its kernel is built or run (patch 0056).

A kernel that assembles a sparse result one entry at a time appends without a check, so a
result past ``2**31 - 1`` entries was refused only after every entry was in memory, or the
process was killed for memory first.  Now each call whose result shape can pass the bound
at a sparse level bounds every such level from the shapes and the operands' stored counts
(scorch/result_bound.py) and refuses with ``index_position_exceeds_int32`` before the
kernel is built (a cold call: after lowering, which says which levels the kernel checks
itself) or run (a warm call).  A level the two-pass assembly checks exactly (patch 0038c's
``scorch_check_level_total``) is left to that check.

1. The bound itself, with no kernel.
2. einsum at the real bound: the refusals, with nothing built and little memory; SpGEMM
   ``ds,ds -> ds`` still computed where the bound passes but the two-pass total does not;
   a warm call refused when only the stored counts grow; a result under the bound computed.
3. Addition, ``to_sparse`` and ``change_mode_order`` with the bound lowered to 1,000 positions (the code reads
   ``result_bound.MAX_INDEX_POSITION``), so that small tensors reach both sides of it.
"""

from __future__ import annotations

import os
import resource
from pathlib import Path
from typing import List

import pytest
import torch

import scorch
from scorch import ops, result_bound, stensor
from scorch.exceptions import (
    IndexCapacityError,
    ScorchAssertionError,
    TensorValidationError,
    defect_code,
)
from scorch.format import MAX_INDEX_POSITION, LevelType
from scorch.result_bound import entry_bound, level_totals_in_source
from scorch.stensor import STensor

CODE = "index_position_exceeds_int32"
D, S, COO = LevelType.DENSE, LevelType.COMPRESSED, LevelType.COORDINATE
N = 50_000  # N * N passes 2**31 - 1


def assert_capacity_error(error: BaseException) -> None:
    assert type(error) is IndexCapacityError
    assert error.code == CODE
    assert isinstance(error, RuntimeError)
    assert isinstance(error, TensorValidationError)


def _kernel_sources() -> List[Path]:
    """Every kernel source written so far: a build writes its main.cpp first
    (tests/conftest.py gives each session its own directory)."""
    return sorted(Path(os.environ["TORCH_EXTENSIONS_DIR"]).rglob("main.cpp"))


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


@pytest.fixture
def no_build(monkeypatch):
    """A build of a generated kernel fails the test."""

    def refuse(*args, **kwargs):
        raise AssertionError("a generated kernel was built")

    monkeypatch.setattr(ops, "_prepare_generated_kernel_build", refuse)


@pytest.fixture
def fresh_caches(monkeypatch):
    monkeypatch.setattr(ops, "_einsum_dispatch_cache", {})
    monkeypatch.setattr(ops, "_kernel_cache", {})
    monkeypatch.setattr(ops, "_kernel_level_totals", {})
    monkeypatch.setattr(stensor, "_RESULT_BOUNDS", {})
    monkeypatch.setattr(stensor, "_GENERATED_KERNELS", {})


def _sparse_vector(n: int, stored: int, seed: int) -> STensor:
    generator = torch.Generator().manual_seed(seed)
    crd = torch.sort(torch.randperm(n, generator=generator)[:stored]).values.to(
        torch.int32
    )
    values = torch.rand(stored, generator=generator, dtype=torch.float64) + 0.5
    pos = torch.tensor([0, stored], dtype=torch.int32)
    return STensor.from_components((n,), "s", [[pos, crd]], values)


def _rows_csr(rows: int, cols: int, row_entries, seed: int) -> STensor:
    """A ds matrix whose row r stores row_entries(r) columns (sorted column lists)."""
    generator = torch.Generator().manual_seed(seed)
    pos, crd = [0], []
    for row in range(rows):
        columns = row_entries(row)
        crd.extend(columns)
        pos.append(len(crd))
    values = torch.rand(len(crd), generator=generator, dtype=torch.float64) + 0.5
    return STensor.from_components(
        (rows, cols),
        "ds",
        [
            [],
            [
                torch.tensor(pos, dtype=torch.int32),
                torch.tensor(crd, dtype=torch.int32),
            ],
        ],
        values,
    )


# --------------------------------------------------------------------------------------
# 1. The bound
# --------------------------------------------------------------------------------------


def test_a_shape_that_cannot_pass_has_no_bound():
    assert entry_bound("t", "ij", (1000, 1000), (D, S), [[(0, "ij")]]) is None
    # No sparse level: the kernel's entry check bounds every dense level.
    assert entry_bound("t", "ij", (N, N), (D, D), [[(0, "ij")]]) is None
    # A sparse level past the bound only below the dense prefix that fits.
    assert entry_bound("t", "ij", (N, 2), (D, S), [[(0, "ij")]]) is None


@pytest.mark.parametrize(
    "names, extents, kinds",
    [("ij", (N, N), (D,)), ("i", (N, N), (D, S)), ("ij", (N,), (D, S))],
)
def test_levels_of_different_lengths_fail_an_internal_check_with_a_code(
    names, extents, kinds
):
    # Every caller builds the three from one rank, so this is an internal check: a
    # ScorchAssertionError (an AssertionError that also runs under python -O) with a code.
    with pytest.raises(ScorchAssertionError) as caught:
        entry_bound("t", names, extents, kinds, [[(0, names)]])
    assert type(caught.value) is ScorchAssertionError
    assert isinstance(caught.value, AssertionError)
    assert caught.value.code == "result_bound_levels_misaligned"
    assert defect_code(caught.value) == "result_bound_levels_misaligned"
    assert f"{len(names)} names, {len(extents)} extents, {len(kinds)} kinds" in str(
        caught.value
    )


def _one_entry_csr(rows: int, cols: int) -> STensor:
    """A ds matrix of this shape storing one entry, at (0, 0)."""
    pos = torch.ones(rows + 1, dtype=torch.int32)
    pos[0] = 0
    return STensor.from_components(
        (rows, cols),
        "ds",
        [[], [pos, torch.zeros(1, dtype=torch.int32)]],
        torch.ones(1, dtype=torch.float64),
    )


def test_the_public_entries_refuse_another_rank_before_the_bound(no_build):
    # The entries whose result can reach the bound, each given a format or operand of
    # another rank: each refuses with its own code before the bound is built (sapling
    # probe 83580 found the same on 14 calls), so the internal check above stays internal.
    big = _one_entry_csr(N, N)
    for fmt in ("s", "sss"):
        with pytest.raises(TensorValidationError) as caught:
            scorch.einsum("ij,jk->ik", big, big, format=fmt)
        assert defect_code(caught.value) == "einsum_output_format_rank_mismatch"
    small = _one_entry_csr(4, 4)
    dense = torch.ones(4, 4, dtype=torch.float64)
    for fmt in ("s", "sss"):
        with pytest.raises(TensorValidationError) as caught:
            STensor.from_torch(dense).to_sparse(fmt)
        assert defect_code(caught.value) == "format_rank_mismatch"
    with pytest.raises(TensorValidationError) as caught:
        small.change_mode_order([1, 0, 2])
    assert defect_code(caught.value) == "mode_order_not_permutation"
    with pytest.raises(TensorValidationError) as caught:
        small + _sparse_vector(4, 2, seed=3)
    assert defect_code(caught.value) == "add_shape_mismatch"


def test_outer_product_bound_is_the_product_of_stored_counts():
    bound = entry_bound("outer", "ij", (N, N), (S, S), [[(0, "i"), (1, "j")]])
    assert bound is not None
    level0, level1 = bound.levels([40_000, 40_000])
    assert (level0.bound, level1.bound) == (40_000, 1_600_000_000)
    bound.refuse_past_limit([40_000, 40_000])  # 1.6e9 entries fit
    with pytest.raises(IndexCapacityError) as caught:
        bound.refuse_past_limit([N, N])
    assert_capacity_error(caught.value)
    assert "result level 1 (compressed) may hold 2500000000 entries" in str(
        caught.value
    )


def test_product_bound_takes_the_tightest_of_its_terms():
    # ij,jk->ik into ss: min(P, nnzA * nnzB, nnzA * n_k, nnzB * n_i) at level 1.
    bound = entry_bound(
        "spgemm", "ik", (100_000, 100_000), (S, S), [[(0, "ij"), (1, "jk")]]
    )
    assert bound is not None
    _, level1 = bound.levels([1_000, 3_000_000])
    assert level1.stored_bound == min(1_000 * 3_000_000, 1_000 * 100_000)
    assert level1.bound == 100_000_000


def test_a_sum_bounds_by_the_sum_of_its_terms():
    bound = entry_bound("add", "ab", (N, N), (D, S), [[(0, "ab")], [(1, "ab")]])
    assert bound is not None
    (level1,) = bound.levels([1_000_000_000, 1_200_000_000])
    assert level1.stored_bound == 2_200_000_000
    with pytest.raises(IndexCapacityError):
        bound.refuse_past_limit([1_000_000_000, 1_200_000_000])
    bound.refuse_past_limit([1_000_000_000, 1_000_000_000])


def test_a_compressed_level_below_takes_one_position_less():
    bound = entry_bound("t", "abc", (N, N, 2), (S, S, S), [[(0, "abc")]])
    assert bound is not None
    limits = {level: limit for level, _, limit in bound.limits}
    assert limits == {
        0: MAX_INDEX_POSITION - 1,
        1: MAX_INDEX_POSITION - 1,
        2: MAX_INDEX_POSITION,
    }
    with pytest.raises(IndexCapacityError):
        bound.refuse_past_limit([MAX_INDEX_POSITION])  # level 1 holds one too many
    bound.refuse_past_limit([MAX_INDEX_POSITION - 1])


def test_a_coordinate_level_is_bounded_by_the_deepest_level_of_its_run():
    bound = entry_bound("t", "ij", (N, N), (COO, COO), [[(0, "i"), (1, "j")]])
    assert bound is not None
    level0, level1 = bound.levels([N, N])
    assert level0.bound == level1.bound == N * N


def test_a_level_the_kernel_checks_itself_is_left_to_that_check():
    bound = entry_bound("t", "ij", (N, N), (D, S), [[(0, "i"), (1, "j")]])
    assert bound is not None
    with pytest.raises(IndexCapacityError):
        bound.refuse_past_limit([N, N])
    bound.with_checked(frozenset({1})).refuse_past_limit([N, N])


def test_the_level_totals_are_read_from_the_source():
    source = (
        "  scorch_check_level_total(_total1, false, 1);\n"
        "  scorch_check_level_total(_total2, true, 2);\n"
        "  scorch_check_drained_entries(n);\n"
    )
    assert level_totals_in_source(source) == frozenset({1, 2})
    assert level_totals_in_source("int main() { return 0; }") == frozenset()


# --------------------------------------------------------------------------------------
# 2. einsum at the real bound
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("result_format", ["ds", "ss", "oo"])
def test_an_outer_product_past_the_bound_is_refused_before_a_build(
    result_format, fresh_caches, no_build
):
    x, y = _sparse_vector(N, N, seed=1), _sparse_vector(N, N, seed=2)
    before, rss = _kernel_sources(), _peak_rss_gb()
    with pytest.raises(IndexCapacityError) as caught:
        scorch.einsum("i,j->ij", x, y, format=result_format)
    assert_capacity_error(caught.value)
    assert "result level" in str(caught.value)
    assert _kernel_sources() == before
    assert _peak_rss_gb() - rss < 0.5


def test_spgemm_into_ds_is_left_to_its_two_pass_total(fresh_caches):
    """A's rows all store column 0 and B's row 0 is empty, so the product stores
    nothing, while the bound (nnz(A) * nnz(B) and the cells) passes the limit.  Into
    ``ds`` the kernel checks level 1's total itself and the result is computed; into
    ``ss`` level 1 is assembled one entry at a time and the call is refused."""
    a = _rows_csr(N, 4, lambda row: [0], seed=3)
    b = _rows_csr(4, N, lambda row: [] if row == 0 else list(range(N)), seed=4)
    result = scorch.einsum("ij,jk->ik", a, b, format="ds")
    assert result.values.numel() == 0
    assert any(
        "scorch_check_level_total(" in path.read_text() for path in _kernel_sources()
    )
    before = _kernel_sources()
    with pytest.raises(IndexCapacityError) as caught:
        scorch.einsum("ij,jk->ik", a, b, format="ss")
    assert_capacity_error(caught.value)
    assert _kernel_sources() == before


def test_a_warm_call_is_refused_when_only_the_stored_counts_grow(fresh_caches):
    small_x, small_y = _sparse_vector(N, 10, seed=5), _sparse_vector(N, 12, seed=6)
    result = scorch.einsum("i,j->ij", small_x, small_y, format="ds")
    expected = torch.outer(
        small_x.to_dense().values.reshape(-1), small_y.to_dense().values.reshape(-1)
    )
    rows = torch.repeat_interleave(
        torch.arange(N), torch.diff(result.storage._mode_indices[1][0].long())
    )
    cols = result.storage._mode_indices[1][1].long()
    assert result.values.numel() == 120
    assert torch.equal(result.values, expected[rows, cols])
    entries, before, rss = (
        len(ops._einsum_dispatch_cache),
        _kernel_sources(),
        _peak_rss_gb(),
    )
    big_x, big_y = _sparse_vector(N, N, seed=7), _sparse_vector(N, N, seed=8)
    with pytest.raises(IndexCapacityError) as caught:
        scorch.einsum("i,j->ij", big_x, big_y, format="ds")
    assert_capacity_error(caught.value)
    assert len(ops._einsum_dispatch_cache) == entries
    assert _kernel_sources() == before
    assert _peak_rss_gb() - rss < 0.5


# --------------------------------------------------------------------------------------
# 3. Addition and to_sparse, with the bound lowered to 1,000 positions
# --------------------------------------------------------------------------------------


@pytest.fixture
def bound_1000(monkeypatch, fresh_caches):
    monkeypatch.setattr(result_bound, "MAX_INDEX_POSITION", 1000)


def _ss(n: int, stored: int, seed: int) -> STensor:
    generator = torch.Generator().manual_seed(seed)
    flat = torch.randperm(n * n, generator=generator)[:stored]
    dense = torch.zeros(n * n, dtype=torch.float64)
    dense[flat] = torch.rand(stored, generator=generator, dtype=torch.float64) + 0.5
    return STensor.from_torch(dense.reshape(n, n)).to_sparse("ss")


def test_addition_is_bounded_by_both_operands_stored_entries(bound_1000, monkeypatch):
    a, b = _ss(100, 400, seed=9), _ss(100, 400, seed=10)
    total = a + b
    assert torch.equal(
        total.to_dense().values, a.to_dense().values + b.to_dense().values
    )
    c, d = _ss(100, 600, seed=11), _ss(100, 600, seed=12)
    monkeypatch.setattr(
        stensor,
        "_generated_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("built")),
    )
    with pytest.raises(IndexCapacityError) as caught:
        c + d
    assert_capacity_error(caught.value)
    assert "addition: result level 1 (compressed) may hold 1200 entries" in str(
        caught.value
    )


def test_to_sparse_from_dense_counts_the_nonzeros_exactly(bound_1000, monkeypatch):
    generator = torch.Generator().manual_seed(13)
    dense = torch.zeros(100 * 100, dtype=torch.float64)
    dense[torch.randperm(10_000, generator=generator)[:500]] = 1.5
    fits = STensor.from_torch(dense.reshape(100, 100)).to_sparse("ds")
    assert fits.values.numel() == 500
    assert torch.equal(
        fits.to_dense().values.reshape(100, 100), dense.reshape(100, 100)
    )
    dense[torch.randperm(10_000, generator=generator)[:1500]] = 2.5
    monkeypatch.setattr(
        stensor,
        "_generated_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("built")),
    )
    with pytest.raises(IndexCapacityError) as caught:
        STensor.from_torch(dense.reshape(100, 100)).to_sparse("ds")
    assert_capacity_error(caught.value)
    assert "to_sparse: result level 1" in str(caught.value)


def test_change_mode_order_is_bounded_by_the_sources_stored_entries(
    fresh_caches, monkeypatch
):
    # Both sources are built under the real bound: to_sparse would refuse 1,200 entries
    # under the lowered one.
    small, large = _ss(100, 800, seed=14), _ss(100, 1200, seed=15)
    monkeypatch.setattr(result_bound, "MAX_INDEX_POSITION", 1000)
    moved = small.copy()
    moved.change_mode_order([1, 0])
    # to_torch densifies its receiver in place, so each side converts a copy.
    assert torch.equal(moved.copy().to_torch(), small.copy().to_torch())
    monkeypatch.setattr(
        stensor,
        "_generated_kernel",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("built")),
    )
    with pytest.raises(IndexCapacityError) as caught:
        large.change_mode_order([1, 0])
    assert_capacity_error(caught.value)
    assert "change_mode_order: result level" in str(caught.value)
