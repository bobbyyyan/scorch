from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import load_inline

_CSRC = Path(__file__).resolve().parents[2] / "src" / "scorch" / "csrc"


@lru_cache(maxsize=1)
def _runtime_helpers():
    source = r"""
#include "header.h"

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
"""
    header_digest = hashlib.sha256(
        (_CSRC / "header.h").read_bytes() + source.encode()
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
