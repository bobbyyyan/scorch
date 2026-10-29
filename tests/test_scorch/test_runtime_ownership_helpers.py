from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load_inline

from tests.test_scorch.allocation_failure_support import run_with_null_allocator

_CSRC = Path(__file__).resolve().parents[2] / "src" / "scorch" / "csrc"


@lru_cache(maxsize=1)
def _runtime_helpers():
    source = r"""
#include "header.h"
#include "workspace_pool.h"

torch::Tensor vector_float_roundtrip() {
  std::vector<float> values{1.25f, -2.5f, 7.0f};
  const float* original = values.data();
  torch::Tensor tensor =
      scorch_tensor_from_vector(std::move(values), torch::kFloat32);
  TORCH_CHECK(tensor.data_ptr<float>() == original,
              "vector-to-Torch transfer copied its storage");
  return tensor;
}

torch::Tensor unique_array_float_roundtrip() {
  auto values = std::unique_ptr<float[]>(new float[3]);
  values[0] = 2.0f;
  values[1] = -4.0f;
  values[2] = 8.0f;
  const float* original = values.get();
  torch::Tensor tensor = scorch_tensor_from_unique_array(
      std::move(values), 3, torch::kFloat32);
  TORCH_CHECK(tensor.data_ptr<float>() == original,
              "unique-array-to-Torch transfer copied its storage");
  return tensor;
}

void vector_dtype_mismatch() {
  std::vector<int> values{1, 2, 3};
  (void)scorch_tensor_from_vector(std::move(values), torch::kFloat32);
}

bool aligned_buffer_is_initialized_and_aligned() {
  auto values = scorch_make_aligned_buffer<float>(4, 64);
  const auto address = reinterpret_cast<uintptr_t>(values.get());
  values.get()[0] = 3.5f;
  values.get()[3] = -1.0f;
  return address % 64 == 0 && values.get()[0] == 3.5f &&
      values.get()[3] == -1.0f;
}

void aligned_buffer_rejects_underalignment() {
  struct alignas(128) OverAlignedPod {
    float value;
  };
  static_assert(std::is_trivial<OverAlignedPod>::value,
                "test type must exercise alignment, not lifetime rejection");
  (void)scorch_make_aligned_buffer<OverAlignedPod>(1, 64);
}

void coo_1d_growth_overflow() {
  coo_workspace_1d<float, 1> workspace(1);
  workspace.insert(std::numeric_limits<int64_t>::max(), 1.0f);
}

void coo_multidimensional_out_of_range() {
  coo_workspace<float, 2> workspace(4, {2, 3});
  workspace.insert({0, 3}, 1.0f);
}

void coo_multidimensional_flatten_overflow() {
  coo_workspace<float, 2> workspace(
      4, {std::numeric_limits<int64_t>::max(), 2});
  workspace.insert({std::numeric_limits<int64_t>::max() - 1, 1}, 1.0f);
}

bool deferred_linked_owner_initializes_on_first_view() {
  std::vector<linked_list_workspace_1d<float>> pool;
  pool.reserve(2);
  pool.emplace_back(8, true);
  pool.emplace_back(8, true);

  // Model consecutive OpenMP regions using different worker ids. The second
  // owner was not touched by phase 1 and must still be safe in phase 3.
  auto phase1 = pool[0].make_view();
  phase1.insert(2, 1.5f);
  phase1.clear();

  auto phase3 = pool[1].make_view();
  phase3.insert(5, 4.0f);
  auto item = *phase3.begin();
  return phase3.size() == 1 && item.first == 5 && item.second == 4.0f;
}

// Item 5 v5: coo_workspace_owner_1d and its worker views.  kUnmappableCoo is 2**50
// elements (8 PiB of doubles and 8 PiB of int64 indices), an allocation no host can map:
// it makes an allocation fail on demand.
constexpr int64_t kUnmappableCoo = int64_t{1} << 50;

// The view as coo_workspace_1d<T, 1>: an insert sums into a coordinate already there,
// iteration is in insertion order, clear() resets only the flags it set, and sort()
// orders the coordinates.  0 when every check holds, else the failing check.
int coo_view_inserts_iterates_and_clears() {
  coo_workspace_owner_1d<double> owner(8);  // a serial kernel's owner: allocated here
  auto view = owner.make_view();
  if (view.failed() || owner.allocation_failed()) return 1;
  view.insert_unchecked(5, 1.0);
  view.insert_unchecked(2, 2.0);
  view.insert_unchecked(5, 0.5);
  std::vector<std::pair<int64_t, double>> seen;
  for (const auto& it : view) seen.push_back(it);
  if (view.size() != 2 || seen.size() != 2) return 2;
  if (seen[0].first != 5 || seen[0].second != 1.5) return 3;
  if (seen[1].first != 2 || seen[1].second != 2.0) return 4;
  view.clear();
  if (view.size() != 0 || view.begin() != view.end()) return 5;
  view.insert_unchecked(2, 3.0);  // the flag was reset: a new entry, not a sum
  if (view.size() != 1 || (*view.begin()).second != 3.0) return 6;
  view.clear();
  auto again = owner.make_view();  // the same arrays, every flag zero again
  again.insert(6, 1.0);
  again.insert(1, 1.0);
  again.insert(4, 1.0);
  again.sort();
  std::vector<int64_t> order;
  for (const auto& it : again) order.push_back(it.first);
  if (order != std::vector<int64_t>{1, 4, 6}) return 7;
  return 0;
}

// The checked insert refuses a coordinate outside [0, capacity), as the linked-list view's.
void coo_view_checked_insert_out_of_range(int64_t coord) {
  coo_workspace_owner_1d<float> owner(8);
  auto view = owner.make_view();
  view.insert(coord, 1.0f);
}

bool coo_region_owner_allocates_nothing_before_its_first_view() {
  std::vector<coo_workspace_owner_1d<double>> unmappable;
  unmappable.reserve(3);
  for (int worker = 0; worker < 3; ++worker) {
    unmappable.emplace_back(kUnmappableCoo, true);  // must not allocate
  }
  std::vector<coo_workspace_owner_1d<double>> pool;
  pool.reserve(2);
  pool.emplace_back(8, true);
  pool.emplace_back(8, true);
  // Two regions on different worker ids: each owner allocates and zero-fills at its own
  // first view, and a later view of the same owner reuses its arrays.
  auto phase1 = pool[0].make_region_view();
  if (phase1.failed()) return false;
  phase1.insert_unchecked(2, 1.5);
  phase1.clear();
  scorch_check_workspace_pool(pool);
  auto phase3 = pool[1].make_region_view();
  if (phase3.failed()) return false;
  phase3.insert_unchecked(5, 4.0);
  phase3.insert_unchecked(5, 0.5);
  auto again = pool[0].make_region_view();
  again.insert_unchecked(2, 2.0);  // phase1 cleared coordinate 2: a new entry
  scorch_check_workspace_pool(pool);
  auto item = *phase3.begin();
  auto other = *again.begin();
  return phase3.size() == 1 && item.first == 5 && item.second == 4.5 &&
      again.size() == 1 && other.first == 2 && other.second == 2.0 &&
      !pool[0].allocation_failed() && !pool[1].allocation_failed();
}

bool coo_region_owner_records_a_failed_allocation() {
  std::vector<coo_workspace_owner_1d<double>> pool;
  pool.reserve(2);
  pool.emplace_back(kUnmappableCoo, true);
  pool.emplace_back(kUnmappableCoo, true);
  auto view = pool[0].make_region_view();  // noexcept: no exception escapes
  auto retry = pool[0].make_region_view();
  return view.failed() && retry.failed() && view.size() == 0 &&
      !(view.begin() != view.end()) && pool[0].allocation_failed() &&
      !pool[1].allocation_failed();
}

void coo_region_pool_check_throws_after_a_failed_allocation() {
  std::vector<coo_workspace_owner_1d<double>> pool;
  pool.reserve(2);
  pool.emplace_back(8, true);
  pool.emplace_back(kUnmappableCoo, true);
  (void)pool[0].make_region_view();
  (void)pool[1].make_region_view();
  scorch_check_workspace_pool(pool);  // std::bad_alloc
}

void coo_region_owner_capacity_past_size_t() {
  coo_workspace_owner_1d<double> owner(std::numeric_limits<int64_t>::max(), true);
}

void coo_eager_owner_capacity_past_size_t() {
  coo_workspace_owner_1d<double> owner(std::numeric_limits<int64_t>::max());
}

void coo_owner_negative_capacity() {
  coo_workspace_owner_1d<double> owner(-1, true);
}
"""
    header_digest = hashlib.sha256(
        (_CSRC / "header.h").read_bytes()
        + (_CSRC / "workspace_pool.h").read_bytes()
        + source.encode()
    ).hexdigest()[:12]
    return load_inline(
        name=f"scorch_runtime_ownership_{header_digest}",
        cpp_sources=source,
        functions=[
            "vector_float_roundtrip",
            "unique_array_float_roundtrip",
            "vector_dtype_mismatch",
            "aligned_buffer_is_initialized_and_aligned",
            "aligned_buffer_rejects_underalignment",
            "coo_1d_growth_overflow",
            "coo_multidimensional_out_of_range",
            "coo_multidimensional_flatten_overflow",
            "deferred_linked_owner_initializes_on_first_view",
            "coo_view_inserts_iterates_and_clears",
            "coo_view_checked_insert_out_of_range",
            "coo_region_owner_allocates_nothing_before_its_first_view",
            "coo_region_owner_records_a_failed_allocation",
            "coo_region_pool_check_throws_after_a_failed_allocation",
            "coo_region_owner_capacity_past_size_t",
            "coo_eager_owner_capacity_past_size_t",
            "coo_owner_negative_capacity",
        ],
        extra_include_paths=[str(_CSRC)],
        extra_cflags=["-O0"],
        verbose=False,
    )


def test_vector_tensor_transfer_is_zero_copy_and_dtype_checked() -> None:
    helpers = _runtime_helpers()
    tensor = helpers.vector_float_roundtrip()

    assert tensor.dtype == torch.float32
    assert tensor.tolist() == [1.25, -2.5, 7.0]
    with pytest.raises(Exception, match="does not match requested Torch dtype"):
        helpers.vector_dtype_mismatch()


def test_unique_array_tensor_transfer_is_zero_copy() -> None:
    tensor = _runtime_helpers().unique_array_float_roundtrip()

    assert tensor.dtype == torch.float32
    assert tensor.tolist() == [2.0, -4.0, 8.0]


def test_aligned_buffer_checks_element_alignment() -> None:
    helpers = _runtime_helpers()

    assert helpers.aligned_buffer_is_initialized_and_aligned()
    with pytest.raises(Exception, match="does not satisfy element alignment"):
        helpers.aligned_buffer_rejects_underalignment()


def test_coo_workspace_rejects_growth_and_flattening_overflow() -> None:
    helpers = _runtime_helpers()

    with pytest.raises(Exception, match="exceeds supported capacity"):
        helpers.coo_1d_growth_overflow()
    with pytest.raises(Exception, match="coordinate is out of range"):
        helpers.coo_multidimensional_out_of_range()
    with pytest.raises(Exception, match="coordinate flattening overflow"):
        helpers.coo_multidimensional_flatten_overflow()


def test_deferred_linked_owner_initializes_each_worker_on_first_use() -> None:
    assert _runtime_helpers().deferred_linked_owner_initializes_on_first_view()


def test_coo_view_inserts_iterates_and_clears_as_coo_workspace_1d() -> None:
    """Item 5 v5: a worker's ``coo_workspace_view_1d`` sums repeated coordinates,
    iterates in insertion order, resets only the flags it set when cleared (so the
    owner's next view starts empty), and sorts on request."""

    assert _runtime_helpers().coo_view_inserts_iterates_and_clears() == 0


@pytest.mark.parametrize("coord", [-1, 8])
def test_coo_view_checked_insert_refuses_a_coordinate_out_of_range(coord: int) -> None:
    with pytest.raises(Exception, match="workspace coordinate is out of range"):
        _runtime_helpers().coo_view_checked_insert_out_of_range(coord)


def test_coo_region_owner_allocates_at_its_first_view() -> None:
    """An owner built to allocate in the region allocates nothing when built (three
    of an unmappable capacity build), and each owner allocates and zero-fills at its
    own first ``make_region_view()``, then reuses those arrays."""

    assert _runtime_helpers().coo_region_owner_allocates_nothing_before_its_first_view()


# The coo helpers whose allocation fails on purpose, in a child process (see
# allocation_failure_support): argv[1] is this file.
_COO_FAILURE_CHILD = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("runtime_ownership_child", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
helpers = module._runtime_helpers()
print("recorded", helpers.coo_region_owner_records_a_failed_allocation())
try:
    helpers.coo_region_pool_check_throws_after_a_failed_allocation()
except RuntimeError as error:
    print("pool check RuntimeError", "std::bad_alloc" in str(error))
else:
    print("pool check no error")
"""


def test_coo_region_owner_failure_is_recorded_and_thrown_after_the_region() -> None:
    """A failed allocation inside a region is a failed view, never an exception (one
    leaving an OpenMP region ends the process); the pool check after the region
    throws std::bad_alloc, as an allocation before the region would have.  It
    reaches Python as the RuntimeError naming it that torch::wrap_pybind_function
    raises: load_inline binds every function through it, a generated kernel's
    ``evaluate`` included."""

    child = run_with_null_allocator(_COO_FAILURE_CHILD, __file__)
    assert child.returncode == 0, (child.returncode, child.stdout, child.stderr[-4000:])
    assert child.stdout.splitlines() == [
        "recorded True",
        "pool check RuntimeError True",
    ], (child.stdout, child.stderr[-4000:])


def test_coo_owner_fails_closed_on_bad_capacities() -> None:
    helpers = _runtime_helpers()
    # A capacity whose byte size passes size_t throws std::bad_array_new_length (a
    # std::bad_alloc, what main's eager ``new T[capacity]`` throws) while the pool
    # is built, in serial code, whether or not the owner allocates in the region.
    # torch's wrapper raises it as a RuntimeError naming it: "bad_array_new_length"
    # under libc++, "std::bad_array_new_length" under libstdc++.
    with pytest.raises(RuntimeError, match="bad_array_new_length"):
        helpers.coo_region_owner_capacity_past_size_t()
    with pytest.raises(RuntimeError, match="bad_array_new_length"):
        helpers.coo_eager_owner_capacity_past_size_t()
    # A negative capacity throws std::invalid_argument, the type and message main's
    # coo_workspace_1d throws; torch's wrapper raises every C++ exception it does not
    # map as a RuntimeError carrying the message.
    with pytest.raises(RuntimeError, match="workspace capacity cannot be negative"):
        helpers.coo_owner_negative_capacity()
