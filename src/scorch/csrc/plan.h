#pragma once

// ---------------------------------------------------------------------------
// SpmmCsrPlan — one CSR x dense product, already resolved, validated and
// unpacked, so that repeating it costs a single Python->C++ hop.
//
// The ordinary dispatch re-derives the same facts on every call: which kernel
// symbol serves this format/dtype pair, whether the tiling selector wants a
// tiled kernel and with which panel widths, that the index arrays are
// structurally sound, and the nested `vector<vector<Tensor>>` the legacy kernel
// ABI expects. For an operand reused across calls -- a graph adjacency in a
// training loop, the archetypal case -- every one of those is a constant, and
// re-deriving them was the whole per-call cost once the redundant Python work
// was removed: a 64x64 SpMM whose kernel runs in 2.4 us was spending ~9.6 us
// getting there.
//
// A plan holds those constants. `run` does O(1) screens and calls the kernel
// core directly, with no shape vectors, no nested tensor vectors, no format
// resolution, no selector consultation and no O(nnz) revalidation.
//
// Two properties matter for correctness:
//
//   * `run` never raises a user-facing error. When anything about the call is
//     outside what this plan was built for -- a different free dimension, a
//     non-contiguous operand, a dtype change, an index array that has been
//     written since -- it returns nullopt and the caller takes the ordinary
//     path, which produces the canonical result or the canonical error message.
//     A plan is an optimization that can always decline, never a second
//     implementation of the contract.
//   * The structure is pinned by the same evidence the ABI memo uses: the
//     source index arrays' data pointers and version counters, recorded at
//     construction and rechecked per call. A plan is therefore no weaker than
//     the memoized validation it replaces, and carries the same documented
//     trade: a write straight through a raw buffer that a tensor aliases does
//     not bump torch's version counter.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <torch/extension.h>

#include "header.h"
#include "native_abi.h"
#include "spmm.h"

namespace scorch_native {

// The kernels the adaptive tiling selector can choose between, plus the typed
// reference kernel that serves float64. Mirrors tiling.py's decision vocabulary
// ("v2" / "tilej" / "tileijk") so the plan is built straight from the memoized
// decision without a translation table.
enum class SpmmPlanKind : int {
  V2 = 0,        // spmm_csr_float_v2, the drop-in float32 kernel
  TileJ = 1,     // spmm_csr_float_tilej, column panels
  TileIJK = 2,   // spmm_csr_float_tileijk, free-dim strips + B relayout
  Reference = 3  // spmm_csr_typed<double>, the float64 route
};

inline bool spmm_plan_kind_from_name(const std::string& name,
                                     SpmmPlanKind& out) {
  if (name == "v2") { out = SpmmPlanKind::V2; return true; }
  if (name == "tilej") { out = SpmmPlanKind::TileJ; return true; }
  if (name == "tileijk") { out = SpmmPlanKind::TileIJK; return true; }
  if (name == "reference") { out = SpmmPlanKind::Reference; return true; }
  return false;
}

// True when `value` is representable in the legacy int32 kernel ABI. A plan is
// refused rather than truncated when it is not, so the ordinary path raises the
// canonical "cannot be represented" error.
inline bool spmm_plan_fits_int32(int64_t value) {
  return value >= 0 && value <= std::numeric_limits<int>::max();
}

class SpmmCsrPlan {
 public:
  SpmmCsrPlan(SpmmPlanKind kind, torch::ScalarType dtype, int64_t rows,
              int64_t cols, int64_t nnz, int64_t free_dim,
              torch::Tensor source_positions, torch::Tensor source_coordinates,
              torch::Tensor positions, torch::Tensor coordinates,
              int64_t tile_size, int64_t panel_free, int64_t panel_contraction)
      : kind_(kind),
        dtype_(dtype),
        rows_(rows),
        cols_(cols),
        nnz_(nnz),
        free_dim_(free_dim),
        source_positions_(std::move(source_positions)),
        source_coordinates_(std::move(source_coordinates)),
        positions_(std::move(positions)),
        coordinates_(std::move(coordinates)),
        tile_size_(tile_size),
        panel_free_(panel_free),
        panel_contraction_(panel_contraction),
        pos_(positions_.data_ptr<int32_t>()),
        crd_(nnz_ > 0 ? coordinates_.data_ptr<int32_t>() : nullptr),
        source_positions_data_(source_positions_.data_ptr()),
        source_coordinates_data_(source_coordinates_.data_ptr()),
        source_positions_version_(abi_version_of(source_positions_)),
        source_coordinates_version_(abi_version_of(source_coordinates_)),
        served_(0) {}

  int64_t rows() const { return rows_; }
  int64_t cols() const { return cols_; }
  int64_t nnz() const { return nnz_; }
  int64_t free_dim() const { return free_dim_; }
  // How many calls this plan has served. Diagnostic, and what the tests assert on
  // to prove the fast path actually fired. A plain counter rather than an atomic
  // because `run` is invoked from Python holding the GIL and does not release it --
  // exactly as the legacy kernel entries do not -- so two threads cannot be inside
  // it at once.
  int64_t served() const { return served_; }
  std::string kind() const {
    switch (kind_) {
      case SpmmPlanKind::V2: return "v2";
      case SpmmPlanKind::TileJ: return "tilej";
      case SpmmPlanKind::TileIJK: return "tileijk";
      case SpmmPlanKind::Reference: return "reference";
    }
    return "unknown";
  }

  // Serves the product, or declines with nullopt. See the header comment: a
  // decline is never an error, it is a handoff to the ordinary path.
  c10::optional<torch::Tensor> run(const torch::Tensor& a_values,
                                   const torch::Tensor& b, int64_t nthreads,
                                   bool atparallel) {
    // -- the operands must be exactly the shape of call this plan was built for
    if (!a_values.defined() || !b.defined()) return c10::nullopt;
    if (a_values.scalar_type() != dtype_ || b.scalar_type() != dtype_) {
      return c10::nullopt;
    }
    if (!a_values.device().is_cpu() || !b.device().is_cpu()) return c10::nullopt;
    if (a_values.layout() != torch::kStrided || b.layout() != torch::kStrided) {
      return c10::nullopt;
    }
    if (a_values.dim() != 1 || a_values.numel() != nnz_) return c10::nullopt;
    if (b.dim() != 2 || b.size(0) != cols_ || b.size(1) != free_dim_) {
      return c10::nullopt;
    }
    if (!a_values.is_contiguous() || !b.is_contiguous()) return c10::nullopt;
    // Lazy views the legacy boundary materializes before reading raw memory.
    if (a_values.is_conj() || a_values.is_neg() || b.is_conj() || b.is_neg()) {
      return c10::nullopt;
    }
    if (nthreads < -1 || nthreads > std::numeric_limits<int>::max()) {
      return c10::nullopt;
    }

    // -- the structure must still be the structure that was validated
    if (source_positions_.data_ptr() != source_positions_data_ ||
        source_coordinates_.data_ptr() != source_coordinates_data_) {
      return c10::nullopt;
    }
    if (abi_version_of(source_positions_) != source_positions_version_ ||
        abi_version_of(source_coordinates_) != source_coordinates_version_) {
      return c10::nullopt;
    }

    const int rows = static_cast<int>(rows_);
    const int free_dim = static_cast<int>(free_dim_);
    const int contraction = static_cast<int>(cols_);
    const int nt = static_cast<int>(nthreads);
    torch::Tensor values;
    switch (kind_) {
      case SpmmPlanKind::V2:
        values = spmm_csr_float_v2_core(
            rows, free_dim, rows, pos_, crd_, a_values.data_ptr<float>(),
            free_dim, b.data_ptr<float>(), static_cast<int>(tile_size_), nt,
            atparallel);
        break;
      case SpmmPlanKind::TileJ:
        values = spmm_csr_float_tilej_core(
            rows, free_dim, rows, pos_, crd_, a_values.data_ptr<float>(),
            contraction, free_dim, b.data_ptr<float>(),
            static_cast<int>(panel_contraction_), nt);
        break;
      case SpmmPlanKind::TileIJK:
        values = spmm_csr_float_tileijk_core(
            rows, free_dim, rows, pos_, crd_, a_values.data_ptr<float>(),
            contraction, b.data_ptr<float>(), static_cast<int>(panel_free_),
            static_cast<int>(panel_contraction_), nt);
        break;
      case SpmmPlanKind::Reference:
        values = spmm_csr_typed_core<double>(
            rows, free_dim, rows, pos_, crd_, a_values.data_ptr<double>(),
            free_dim, b.data_ptr<double>(), static_cast<int>(tile_size_));
        break;
      default:
        return c10::nullopt;
    }
    ++served_;
    // The kernels return a flat contiguous [rows * free_dim] buffer; the
    // ordinary path reshapes it to 2-D in Python. Do it here, so the caller
    // returns what `run` hands back with no further work. `view` cannot copy:
    // the buffer was just allocated contiguous.
    return values.view({rows_, free_dim_});
  }

 private:
  SpmmPlanKind kind_;
  torch::ScalarType dtype_;
  int64_t rows_;
  int64_t cols_;
  int64_t nnz_;
  int64_t free_dim_;
  // The caller's arrays, held so their identity and version can be rechecked.
  // When they were already int32 these are the same tensors as positions_ /
  // coordinates_; when they were int64 they are the originals and the narrowed
  // copies are separate.
  torch::Tensor source_positions_;
  torch::Tensor source_coordinates_;
  torch::Tensor positions_;
  torch::Tensor coordinates_;
  int64_t tile_size_;
  int64_t panel_free_;
  int64_t panel_contraction_;
  const int32_t* pos_;
  const int32_t* crd_;
  const void* source_positions_data_;
  const void* source_coordinates_data_;
  uint32_t source_positions_version_;
  uint32_t source_coordinates_version_;
  int64_t served_;
};

// Builds a plan, or returns nullopt when the configuration is outside what a
// plan can serve (unsupported kernel/dtype pair, extents the legacy int32 ABI
// cannot express, a tiled kind whose kernel requires a positive extent). The
// caller keeps using the ordinary path in that case; nothing here decides
// whether a plan is *wanted*, only whether one is *possible*.
//
// Structural validation runs exactly once, here, through the same
// `checked_csr_view` every kernel entry uses -- so a plan cannot be built over
// an index structure that would have been rejected, and the narrowing memo is
// consulted rather than duplicated.
inline c10::optional<SpmmCsrPlan> make_spmm_csr_plan(
    const std::string& kind_name, std::vector<int64_t> a_shape,
    std::vector<std::vector<torch::Tensor>> a_mode_indices,
    const torch::Tensor& a_values, int64_t free_dim, int64_t panel_free,
    int64_t panel_contraction) {
  SpmmPlanKind kind;
  if (!spmm_plan_kind_from_name(kind_name, kind)) return c10::nullopt;
  if (!a_values.defined()) return c10::nullopt;

  const torch::ScalarType dtype = a_values.scalar_type();
  if (kind == SpmmPlanKind::Reference) {
    if (dtype != torch::kFloat64) return c10::nullopt;
  } else if (dtype != torch::kFloat32) {
    return c10::nullopt;
  }
  if (a_shape.size() != 2) return c10::nullopt;
  if (!spmm_plan_fits_int32(a_shape[0]) || !spmm_plan_fits_int32(a_shape[1]) ||
      !spmm_plan_fits_int32(free_dim)) {
    return c10::nullopt;
  }
  if (a_mode_indices.size() != 2 || a_mode_indices[1].size() != 2) {
    return c10::nullopt;
  }
  if (panel_free < 0 || panel_contraction < 0) return c10::nullopt;
  if (!spmm_plan_fits_int32(panel_free) ||
      !spmm_plan_fits_int32(panel_contraction)) {
    return c10::nullopt;
  }
  // The tiled kernels reject a zero contraction or free extent at their
  // boundary; a plan for such a shape would have to reproduce that error, so
  // decline instead and let the boundary raise it.
  if (kind == SpmmPlanKind::TileJ && a_shape[1] <= 0) return c10::nullopt;
  if (kind == SpmmPlanKind::TileIJK && (a_shape[1] <= 0 || free_dim <= 0)) {
    return c10::nullopt;
  }

  const torch::Tensor source_positions = a_mode_indices[1][0];
  const torch::Tensor source_coordinates = a_mode_indices[1][1];
  const CsrInputView view = checked_csr_view(
      a_shape, a_mode_indices, a_values, dtype, "matmul", "A");
  if (!spmm_plan_fits_int32(view.nnz)) return c10::nullopt;

  // Tile parameters, clamped exactly as the kernel boundaries clamp them, so a
  // planned call and an ordinary call reach the kernel with identical numbers.
  // Each boundary applies validate_tile_size(default, extent=N), and the defaults
  // differ by symbol: 256 for v2, 32 for the typed reference kernel (which ignores
  // the value, but matching it costs nothing and removes a trap). The tiled kernels
  // clamp their panel widths internally and take them as given here.
  const int64_t tile_default = kind == SpmmPlanKind::Reference ? 32 : 256;
  const int64_t tile_size =
      std::min<int64_t>(tile_default, std::max<int64_t>(free_dim, 1));
  return SpmmCsrPlan(kind, dtype, a_shape[0], a_shape[1], view.nnz, free_dim,
                     source_positions, source_coordinates, view.positions,
                     view.coordinates, tile_size, panel_free,
                     panel_contraction);
}

}  // namespace scorch_native
