#pragma once

// What scorch hands back when every input to a product was a torch tensor: a torch tensor in the
// layout torch itself returns for that call (the drop-in's result-type rule (a)), built from the
// arrays the kernel wrote, with no copy.
//
// One place for this, so that a product has one result type whichever route computed it: a CSR x
// CSR product reaches it through spgemm_csr_torch_result (ops.cpp), which the C entry point
// (planned_matmul.h) and the Python function (ops.py, _torch_csr_spgemm) both call.

#include <ATen/ATen.h>

namespace scorch_native {

// The 2-D torch sparse CSR tensor over `positions`, `coordinates` and `values`, which it holds by
// reference: no element is copied, and each array keeps its own storage, deleter included. That
// matters for a values buffer some other owner reclaims from its storage's deleter (a plan's output
// slot): the buffer goes back only when the last tensor over it -- this result's values among them
// -- is gone. A data pointer re-wrapped with from_blob and a default deleter would hand it back
// while the result still pointed at it.
//
// Unchecked: torch's unsafe constructor runs no invariant check, because the callers pass a
// kernel's own output, whose positions and coordinates are of one index dtype (int32 or int64) and
// whose rows are sorted and distinct by construction (spmspm_csr_core). The tests hold every route
// to torch._validate_sparse_csr_tensor_args, so that a kernel that broke this would be caught
// there rather than by torch's mul or sparse_mask computing silently wrong values from it.
inline at::Tensor csr_result_as_torch(const at::Tensor& positions, const at::Tensor& coordinates,
                                      const at::Tensor& values, int64_t rows, int64_t cols) {
  return at::_sparse_csr_tensor_unsafe(positions, coordinates, values, {rows, cols},
                                       values.options().layout(c10::kSparseCsr));
}

}  // namespace scorch_native
