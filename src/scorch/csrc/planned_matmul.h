#pragma once

// scorch.matmul's planned fast path, in C.
//
// A repeated CSR x dense product is served by a plan (plan.h, plan.py): everything the
// ordinary dispatch would re-derive is a constant of the operand and the free dimension, so
// a planned call is the plan's checks and the kernel. What was left in front of the plan was
// the Python that finds it -- the call into a large Python function, two type tests, a key
// tuple built from b.shape and b.dtype, two dict probes, a two-level attribute read for the
// values, a call to torch.get_num_threads(), and pybind11's argument conversion into `run`.
// Warm that is a few hundred nanoseconds; cold -- after other layers have evicted the
// interpreter's objects and code -- it is a chain of dependent misses: on mkt1 a planned call
// through scorch.matmul cost 1.2-1.5 us MORE than the legacy entry point it replaced, at
// every width and on both collections, while the native plan alone costs less than either.
//
// So scorch.matmul is this function when the extension provides it. It serves exactly the
// calls the Python probe in ops.matmul served -- two positional operands, no keyword
// arguments, `type(a) is STensor`, `type(b) is torch.Tensor`, plans enabled, a plan stored
// under the same key -- and hands EVERY other call, unchanged and with the same arguments, to
// the Python function. That includes a plan that declines: the Python function then probes
// again, the plan declines again, and plan.py's decline accounting and withdrawal run exactly
// as before. Nothing about installation moves here; this only reads what plan.py wrote.
//
// The key is built to be EQUAL to the Python one, not merely analogous: ((rows, cols), dtype,
// generation). b.shape is a torch.Size, a tuple subclass that inherits tuple's hash and
// equality, so a plain tuple of the same ints finds the same entry; the dtype object is the
// interned one torch hands out for that scalar type; the generation is read from the same
// list cell the Python probe reads.
//
// fx tracing still works because tracing never reaches here: a Proxy is not an STensor, so
// the call goes to the Python function, which records `matmul` -- its module global, which
// is this object -- as the node's target. trace.py's `node.target is ops.matmul` therefore
// still holds.
//
// The left operand may also be a torch sparse CSR tensor (`type(a) is torch.Tensor` with the
// sparse CSR layout), whose plans live in the entry of its STRUCTURE -- its index arrays and
// shape -- in torch_structures.h, shared by every tensor over the same arrays and dropped with
// them, because a torch tensor cannot carry them (plan.py) and they are facts about the arrays.
// Its values and index arrays are read from the tensor's implementation, so a served call
// creates no Python object, and the index arrays go to `run_csr`, which declines unless they
// are still the ones the plan was built from. The layout test is a field read on the
// TensorImpl, so a dense left operand -- every dense matmul made through `import scorch as
// torch` -- pays one comparison for it before going to the Python function as before.
//
// For a torch operand this path also owns the SIGHTINGS (plan.install's rule that a plan is
// installed only for an operand seen before). An operand built afresh for every call is only
// ever sighted, never planned, so whatever a sighting costs is what such a call site pays for
// plans existing at all: done in Python at the end of the ordinary path it was 2 us of a mkt1
// call. So an unserved torch operand goes to the Python function with the key under a private
// keyword (ops._C_PROBED_KW), which tells it the probe was done here and the sighting will be
// recorded here, once it returns. Python keeps installation and decline accounting.

#include "plan.h"
#include "torch_structures.h"

#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorImpl.h>
#include <ATen/FuncTorchTLS.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/object_ptr.h>

#include <atomic>
#include <string>

namespace scorch_native {

struct PlannedMatmul {
  PyObject* ordinary = nullptr;      // the Python matmul: every call a plan does not serve
  PyObject* stensor_type = nullptr;  // scorch.STensor, compared by identity like the probe
  PyObject* tensor_type = nullptr;   // torch.Tensor, likewise (a Parameter is not served)
  PyObject* generation = nullptr;    // plan.GENERATION, a one-element list
  PyObject* enabled = nullptr;       // plan.ENABLED_CELL, a one-element list
  PyObject* plans_attr = nullptr;    // interned plan.PLANS_ATTR
  PyObject* storage_attr = nullptr;  // interned "_storage"
  PyObject* value_attr = nullptr;    // interned "_value"
  // Interned "_metadata" and "requires_grad", for planned_matmul_stensor_wants_grad: static, so
  // they take no place in the state and no member's offset moves.
  static inline PyObject* metadata_attr = nullptr;
  static inline PyObject* requires_grad_attr = nullptr;
  // Torch sparse CSR operands: their plans and sightings live in their structure's entry
  // (torch_structures.h).
  PyObject* seen_attr = nullptr;         // interned plan.SEEN_ATTR
  PyObject* probed_kwnames = nullptr;    // (ops._C_PROBED_KW,): the keyword the key travels under
  Py_ssize_t max_plans = 0;              // plan.MAX_PLANS_PER_OPERAND
  bool match_host_threads = false;       // ops._MATCH_HOST_THREADS
  bool atparallel = false;               // ops._PLAN_ATPARALLEL
  // autograd.planned_mm_backward: the backward of a product PlannedMatmulGrad recorded. Unset,
  // every product autograd must record goes to the Python function.
  PyObject* mm_backward = nullptr;
  // x @ A for a torch CSR A (see planned_matmul_torch_right): autograd.planned_mm_t_backward, the
  // backward of a product PlannedMatmulGradT recorded, and the interned name under which a
  // structure's state holds its TransposedProductPlans (plan.TRANSPOSED_PLANS_ATTR). Unset, every
  // x @ A goes to the Python function.
  PyObject* mm_t_backward = nullptr;
  PyObject* tplans_attr = nullptr;
  // A torch CSR operand's product no plan serves is made here when it is the call the Python
  // function would make (planned_matmul_torch_ordinary). `ordinary_spmm` is the drop-in SpMM at
  // float32 and float64, as ops.cpp calls it for this path; `serve_ordinary` says, per dtype,
  // whether ops.py found the CSR x dense product resolves to it; `tiling_candidate` is
  // tiling.shape_is_candidate, the tiling selector's gate, which must decline a float32 product;
  // `screen_grain` is the construction-time screen's (storage._WRAP_GRAIN). Unset, every such
  // product goes to the Python function.
  using OrdinarySpmm = torch::Tensor (*)(const torch::Tensor& positions,
                                         const torch::Tensor& coordinates,
                                         const torch::Tensor& values, const torch::Tensor& b,
                                         int64_t rows, int64_t cols, int64_t nthreads,
                                         bool atparallel);
  OrdinarySpmm ordinary_spmm[2] = {nullptr, nullptr};
  bool serve_ordinary[2] = {false, false};
  PyObject* tiling_candidate = nullptr;
  int64_t screen_grain = 65536;
  // Products made by planned_matmul_torch_ordinary, for tests: a statistic, which calls on the
  // shared state count through a const reference like every other read of it.
  mutable int64_t ordinary_served = 0;
  // scheduler._SCHEDULE_FORCE, the context variable `schedule_force` sets. While it holds a
  // schedule, every call goes to the Python function, whose generated kernel honours the
  // schedule or refuses it with a code: every product made here is a prebuilt kernel's, which
  // takes no schedule and would ignore it (planned_matmul_schedule_forced). Unset, no call is
  // treated as forced -- a build whose ops.py does not set it never served a forced schedule
  // correctly either.
  PyObject* schedule_force = nullptr;
  // Set by the first schedule_force in the process, before the context variable is, and never
  // cleared (scheduler.FORCE_SEEN_CELL; ops.py registers note_planned_matmul_schedule_force).
  // Until then no context holds a schedule, so the variable is not read: a call pays one load.
  // Relaxed: whatever hands a context to another thread (a queue, a lock, a thread's start)
  // orders this store before that thread's calls.
  std::atomic<bool> schedule_force_seen{false};
  // The product of two torch CSR operands is made here when it is the call the Python function
  // would make (planned_matmul_torch_spgemm). `ordinary_spgemm` is ops.cpp's
  // spgemm_csr_torch_result, the one function both routes call; `serve_spgemm` says, per value
  // dtype (float32, float64, int32, int64), whether ops.py found the CSR x CSR product resolves to
  // the prebuilt kernel; `spgemm_int64_indices` is the result's index type, which ops.py passes to
  // the same function. Unset, every such product goes to the Python function.
  using OrdinarySpgemm = at::Tensor (*)(const at::Tensor& a_positions,
                                        const at::Tensor& a_coordinates,
                                        const at::Tensor& a_values,
                                        const at::Tensor& b_positions,
                                        const at::Tensor& b_coordinates,
                                        const at::Tensor& b_values, int64_t rows, int64_t inner,
                                        int64_t cols, bool int64_indices);
  OrdinarySpgemm ordinary_spgemm = nullptr;
  bool serve_spgemm[4] = {false, false, false, false};
  bool spgemm_int64_indices = false;
  // Products made by planned_matmul_torch_spgemm, for tests, like ordinary_served.
  mutable int64_t spgemm_served = 0;
};

// Whether a schedule is forced in the calling context: no, without a read, until
// schedule_force has run in this process (schedule_force_seen); then one PyContextVar_Get,
// which CPython answers from the variable's per-thread cache while the context is unchanged.
// A read that fails counts as forced, so the Python function, which reads the variable
// itself, decides.
inline bool planned_matmul_schedule_forced(const PlannedMatmul& s) {
  if (!s.schedule_force_seen.load(std::memory_order_relaxed)) return false;
  if (s.schedule_force == nullptr) return false;
  PyObject* value = nullptr;
  if (PyContextVar_Get(s.schedule_force, Py_None, &value) < 0) {
    PyErr_Clear();
    return true;
  }
  const bool forced = value != Py_None;
  Py_DECREF(value);
  return forced;
}

// The slot of serve_spgemm for a value dtype, or -1 when no prebuilt CSR x CSR kernel takes it.
inline int planned_matmul_spgemm_slot(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat: return 0;
    case at::kDouble: return 1;
    case at::kInt: return 2;
    case at::kLong: return 3;
    default: return -1;
  }
}

inline PlannedMatmul& planned_matmul_state() {
  static PlannedMatmul state;
  return state;
}

// The index arrays of a torch sparse operand as every cache must read them, recording `a` as a
// witness of their storages (native_abi.h, Write witnesses).
//
// Not crow_indices()/col_indices()/indices(): those return aliases that share `a`'s version
// counter, so a cache keyed on them saw writes through `a`'s accessors and missed every write
// through the caller's own arrays -- a product after `col.copy_(...)` came back computed from the
// old columns wherever an int32 copy or a transpose of them was cached. These are the members,
// which ARE the caller's arrays (torch keeps the TensorImpls it was given), detached: a new
// TensorImpl over the same storage, sharing their version counter, so writes through the caller's
// arrays and every view of them move it. Writes through `a`'s accessors move `a`'s counter, which
// the witness registered here puts into the storages' epochs.
//
// A CSR operand gives (crow, col), a CSC operand (ccol, row) -- the same impl, its compressed and
// plain indices -- and a COO operand (indices,). Anything else is a TypeError.
inline py::tuple torch_sparse_index_members(const at::Tensor& a) {
  if (a.layout() == c10::kSparseCsr || a.layout() == c10::kSparseCsc) {
    auto* impl = at::sparse_csr::get_sparse_csr_impl(a);
    const at::Tensor& positions = impl->compressed_indices();
    const at::Tensor& coordinates = impl->plain_indices();
    abi_watch_register(positions, a);
    abi_watch_register(coordinates, a);
    return py::make_tuple(positions.detach(), coordinates.detach());
  }
  if (a.layout() == c10::kSparse) {
    auto* impl = static_cast<at::SparseTensorImpl*>(a.unsafeGetTensorImpl());
    const at::Tensor& indices = impl->indices();
    abi_watch_register(indices, a);
    return py::make_tuple(indices.detach());
  }
  throw py::type_error("torch_sparse_index_members expects a sparse CSR, CSC or COO tensor");
}

// The key a plan for a dense operand of these sizes and dtype is stored under, (sizes, dtype,
// generation), as a new reference; nullptr with an error set if it cannot be built.
inline PyObject* planned_matmul_key_sizes(const PlannedMatmul& s, c10::IntArrayRef sizes,
                                          at::ScalarType dtype) {
  if (!PyList_CheckExact(s.generation) || PyList_GET_SIZE(s.generation) < 1) {
    PyErr_SetString(PyExc_RuntimeError, "plan generation cell is malformed");
    return nullptr;
  }
  const int64_t dim = (int64_t)sizes.size();
  PyObject* shape = PyTuple_New(dim);
  if (shape == nullptr) return nullptr;
  for (int64_t d = 0; d < dim; ++d) {
    PyObject* extent = PyLong_FromLongLong(sizes[d]);
    if (extent == nullptr) {
      Py_DECREF(shape);
      return nullptr;
    }
    PyTuple_SET_ITEM(shape, d, extent);
  }
  PyObject* key = PyTuple_Pack(3, shape, reinterpret_cast<PyObject*>(torch::getTHPDtype(dtype)),
                               PyList_GET_ITEM(s.generation, 0));
  Py_DECREF(shape);
  return key;
}

// The key a plan for `a @ bt` is stored under, ((b rows, b cols), b dtype, generation).
inline PyObject* planned_matmul_key(const PlannedMatmul& s, const at::Tensor& bt) {
  return planned_matmul_key_sizes(s, bt.sizes(), bt.scalar_type());
}

// The same key as plain data (torch_structures.h, TorchSighting); false if it has no such form.
inline bool planned_matmul_sighting(const PlannedMatmul& s, const at::Tensor& bt,
                                    TorchSighting& out) {
  if (bt.dim() < 1 || bt.dim() > 2 || !PyList_CheckExact(s.generation) ||
      PyList_GET_SIZE(s.generation) < 1)
    return false;
  const long long generation = PyLong_AsLongLong(PyList_GET_ITEM(s.generation, 0));
  if (generation == -1 && PyErr_Occurred()) {
    PyErr_Clear();
    return false;
  }
  out.dim = bt.dim();
  out.extent[0] = bt.size(0);
  out.extent[1] = bt.dim() == 2 ? bt.size(1) : 0;
  out.dtype = static_cast<int32_t>(bt.scalar_type());
  out.generation = generation;
  return true;
}

inline SpmmCsrPlan* planned_matmul_cast(PyObject* plan_obj) {
  try {
    return py::handle(plan_obj).cast<SpmmCsrPlan*>();
  } catch (const py::cast_error&) {
    return nullptr;
  }
}

// The plan stored in `plans` under `key`, or nullptr with no error set when there is none.
inline SpmmCsrPlan* planned_matmul_find(PyObject* plans, PyObject* key) {
  if (plans == nullptr || !PyDict_CheckExact(plans)) return nullptr;
  PyObject* plan_obj = PyDict_GetItemWithError(plans, key);  // borrowed
  if (plan_obj == nullptr || plan_obj == Py_None) {
    PyErr_Clear();
    return nullptr;
  }
  return planned_matmul_cast(plan_obj);
}

// ---------------------------------------------------------------------------
// x @ A for a torch CSR A
//
// autograd.py computes x @ A as (A^T @ x^T)^T on A's transposed structure, whose product is planned
// like any other: an STensor over the transpose's index arrays (the carrier), whose values are A's,
// carried across by a permutation and refilled on every call. Through the Python function that is
// the whole path on every call, and where a gradient is wanted a Python autograd Function records
// it. On an M5, a cold training step through x @ A on a matrix of under 600 nonzeros took a median
// 34.8 us more than one through A @ B, which this file records (36 cells; torch's two differ by
// 3.1 us), and was slower than torch.mm's in 52 of 1200 cold readings, where A @ B was in none.
//
// So once the transposed product has a plan, x @ A is served, and recorded, here. The plan is found
// through the state of the operand's structure (torch_structures.h): whenever
// planned_matmul_gathered serves the transposed product of a torch CSR operand (its `owner`), it
// records in that state a TransposedProductPlan for it -- the carrier's plan, the permutation, and
// the index arrays the transpose was built from -- under the key that product's plan is stored
// under, so every tensor over the same index arrays finds it.
// A call is served only while the arrays are as they were then: the same buffers and lengths, and
// no write to them since. Two counters see such writes, and neither sees both kinds (native_abi.h,
// Write witnesses): the index arrays' own, which writes through the caller's arrays and their
// views move, and each operand's, shared with every alias its crow_indices(), col_indices() and
// values() return. The record compares the arrays' counters and the epochs of their storages,
// which move on a write through the accessors of any torch sparse tensor scorch has read over
// them -- every operand that reaches this record is one. autograd.py's structure cache compares
// the same, so a call this declines is one whose structure and transpose that cache rebuilds. The
// values are read from the operand on every call, so a write to them needs no check.
//
// The transposed values are gathered into a buffer of the record's own, not the carrier's. The
// carrier's buffer belongs to autograd.py's structure, shared by every operand over the same index
// arrays, and its writers serialize on that structure's lock, which they hold across gather_values
// with the GIL released; a writer here holds only the GIL. So the record's buffer is written only
// here, with the GIL held from the fill to the end of the product (plan runs do not release it),
// and an operand's records of one transpose share one buffer.
struct TransposedProductPlan {
  TransposedProductPlan(py::object plan, SpmmCsrPlan* product, torch::Tensor buffer,
                        torch::Tensor perm, const torch::Tensor& positions,
                        const torch::Tensor& coordinates)
      : plan_(std::move(plan)),
        product_(product),
        buffer_(std::move(buffer)),
        perm_(std::move(perm)),
        positions_(positions),
        coordinates_(coordinates),
        positions_watch_(positions),
        coordinates_watch_(coordinates) {}

  // Whether an operand whose members are these index arrays is over the arrays this was recorded
  // from, unchanged since. It borrows them (BorrowedArray), as the operand's plans do: it lives in
  // their structure's entry, which a reference to them would keep alive forever.
  bool built_from(const at::Tensor& positions, const at::Tensor& coordinates) const {
    return positions_.is(positions) && coordinates_.is(coordinates) &&
           positions_watch_.current() && coordinates_watch_.current();
  }

  // Whether this records exactly this carrier plan and permutation.
  bool records(const SpmmCsrPlan* product, const at::Tensor& perm) const {
    return product == product_ && perm.is_same(perm_);
  }

  // The buffer this gathers into, for another record of the same transpose (same permutation)
  // and dtype to share; undefined otherwise.
  torch::Tensor buffer_for(const at::Tensor& perm, at::ScalarType dtype) const {
    return perm.is_same(perm_) && buffer_.scalar_type() == dtype ? buffer_ : torch::Tensor();
  }

  // A^T @ m, A's values being `values`: SpmmCsrPlan::run_gathered on the transpose's plan, which
  // leaves this record's buffer holding A^T's values. nullopt where that declines.
  c10::optional<torch::Tensor> run(const at::Tensor& values, const at::Tensor& m,
                                   int64_t nthreads, bool atparallel) {
    c10::optional<torch::Tensor> out =
        product_->run_gathered(buffer_, values, perm_, m, nthreads, atparallel);
    if (out.has_value()) ++served_;
    return out;
  }

  int64_t served() const { return served_; }
  int64_t free_dim() const { return product_->free_dim(); }
  torch::Tensor buffer() const { return buffer_; }  // for tests

 private:
  py::object plan_;  // the carrier's plan object, which product_ points into: kept alive here
  SpmmCsrPlan* product_;
  torch::Tensor buffer_;
  torch::Tensor perm_;
  BorrowedArray positions_;
  BorrowedArray coordinates_;
  PlanSourceWatch positions_watch_;
  PlanSourceWatch coordinates_watch_;
  int64_t served_ = 0;
};

// The TransposedProductPlan stored in `held` under `key`, or nullptr with no error set.
inline TransposedProductPlan* planned_matmul_find_transposed(PyObject* held, PyObject* key) {
  if (held == nullptr || !PyDict_CheckExact(held)) return nullptr;
  PyObject* obj = PyDict_GetItemWithError(held, key);  // borrowed
  if (obj == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  try {
    return py::handle(obj).cast<TransposedProductPlan*>();
  } catch (const py::cast_error&) {
    return nullptr;
  }
}

// Defined with the torch operands below; called by planned_matmul_gathered.
inline void planned_matmul_record_transposed(const PlannedMatmul& s, py::handle owner,
                                             PyObject* key, PyObject* plan_obj,
                                             SpmmCsrPlan* plan, const torch::Tensor& buffer,
                                             const torch::Tensor& perm);

inline int64_t planned_matmul_nthreads(const PlannedMatmul& s) {
  return s.match_host_threads ? (int64_t)at::get_num_threads() : -1;
}

// Whether a gradient is wanted through the STensor `a`: grad mode is on and its requires_grad
// is set, read as the property reads it (a._metadata.requires_grad). Such a call is the Python
// function's, which records it (scorch/autograd.py); a plan would serve it and the result would
// carry no gradient. Under no_grad the operand's plan still serves it, so a trained sparse
// weight evaluated for inference keeps its planned path. Asked only of a call a plan would
// serve, and grad mode first, so a call under no_grad pays one thread-local read for it and a
// call without a plan nothing. A flag that cannot be read, or is anything but False, counts as
// wanting one: the Python function then decides. Out of line, so that planned_matmul_try, which
// every planned call runs, gains only this call and its branch. Not cold: a call in grad mode,
// torch's default, makes it on every served call, and a cold callee would mark the rest of the
// served path unlikely.
SCORCH_NOINLINE inline bool planned_matmul_stensor_wants_grad(const PlannedMatmul& s,
                                                              PyObject* a) {
  if (!at::GradMode::is_enabled()) return false;
  PyObject* metadata = PyObject_GetAttr(a, s.metadata_attr);
  if (metadata == nullptr) {
    PyErr_Clear();
    return true;
  }
  PyObject* flag = PyObject_GetAttr(metadata, s.requires_grad_attr);
  Py_DECREF(metadata);
  if (flag == nullptr) {
    PyErr_Clear();
    return true;
  }
  const bool wants = flag != Py_False;
  Py_DECREF(flag);
  return wants;
}

// An STensor operand. The served result as a new reference; nullptr with no Python error set
// when this call is not one a plan serves (the caller then runs the ordinary path); nullptr
// WITH an error set only for a failure the ordinary path would also have raised from the same
// call to `run`.
inline PyObject* planned_matmul_try(const PlannedMatmul& s, PyObject* a, PyObject* b) {
  PyObject* dict = PyObject_GenericGetDict(a, nullptr);
  if (dict == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  PyObject* plans = PyDict_GetItemWithError(dict, s.plans_attr);  // borrowed; `a` owns dict
  Py_DECREF(dict);
  if (plans == nullptr || !PyDict_CheckExact(plans)) {
    PyErr_Clear();
    return nullptr;
  }

  const at::Tensor& bt = THPVariable_Unpack(b);
  PyObject* key = planned_matmul_key(s, bt);
  if (key == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  SpmmCsrPlan* plan = planned_matmul_find(plans, key);
  Py_DECREF(key);
  if (plan == nullptr) return nullptr;
  if (planned_matmul_stensor_wants_grad(s, a)) return nullptr;

  // a._storage._value, read the way the probe reads it, so a storage that computes its
  // values on access is honoured. Any failure falls through: the ordinary path raises it.
  PyObject* storage = PyObject_GetAttr(a, s.storage_attr);
  if (storage == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  PyObject* values = PyObject_GetAttr(storage, s.value_attr);
  Py_DECREF(storage);
  if (values == nullptr) {
    PyErr_Clear();
    return nullptr;
  }
  if (!THPVariable_Check(values)) {
    Py_DECREF(values);
    return nullptr;
  }
  c10::optional<torch::Tensor> served;
  try {
    served = plan->run(THPVariable_Unpack(values), bt, planned_matmul_nthreads(s), s.atparallel);
  } catch (...) {
    Py_DECREF(values);
    throw;  // translated to the Python exception by the caller's HANDLE_TH_ERRORS
  }
  Py_DECREF(values);
  if (!served.has_value()) return nullptr;
  return THPVariable_Wrap(std::move(*served));
}

// The product of an STensor `a` whose values are src[perm[q]] -- autograd.py's carrier for a CSR
// transpose -- with `b`, by the plan `a` holds for `b`: SpmmCsrPlan::run_gathered, which leaves
// `a`'s values filled. None when no plan serves it (none yet for this `b`, plans disabled, or a
// decline); the caller then fills the values itself and runs the ordinary path, which plans a
// product it sees again. `owner`, when it is the torch CSR operand whose transpose `a` carries,
// is given a TransposedProductPlan for a served product, so that its x @ A is served here next.
inline py::object planned_matmul_gathered(py::handle a, const torch::Tensor& b,
                                          const torch::Tensor& src, const torch::Tensor& perm,
                                          py::handle owner) {
  const PlannedMatmul& s = planned_matmul_state();
  if (s.plans_attr == nullptr || s.storage_attr == nullptr || s.value_attr == nullptr ||
      !PyList_CheckExact(s.enabled) || PyList_GET_SIZE(s.enabled) < 1 ||
      PyList_GET_ITEM(s.enabled, 0) != Py_True)
    return py::none();
  py::object dict = py::reinterpret_steal<py::object>(PyObject_GenericGetDict(a.ptr(), nullptr));
  if (!dict) {
    PyErr_Clear();
    return py::none();
  }
  PyObject* plans = PyDict_GetItemWithError(dict.ptr(), s.plans_attr);  // borrowed
  py::object key = py::reinterpret_steal<py::object>(
      plans != nullptr ? planned_matmul_key(s, b) : nullptr);
  SpmmCsrPlan* plan = key ? planned_matmul_find(plans, key.ptr()) : nullptr;
  py::object storage = py::reinterpret_steal<py::object>(
      plan != nullptr ? PyObject_GetAttr(a.ptr(), s.storage_attr) : nullptr);
  py::object values = py::reinterpret_steal<py::object>(
      storage ? PyObject_GetAttr(storage.ptr(), s.value_attr) : nullptr);
  PyErr_Clear();
  if (!values || !THPVariable_Check(values.ptr())) return py::none();
  const torch::Tensor& buffer = THPVariable_Unpack(values.ptr());
  c10::optional<torch::Tensor> served =
      plan->run_gathered(buffer, src, perm, b, planned_matmul_nthreads(s), s.atparallel);
  if (!served.has_value()) return py::none();
  if (!owner.is_none())
    planned_matmul_record_transposed(s, owner, key.ptr(),
                                     PyDict_GetItemWithError(plans, key.ptr()), plan, buffer, perm);
  return py::reinterpret_steal<py::object>(THPVariable_Wrap(std::move(*served)));
}

// ---------------------------------------------------------------------------
// Torch sparse CSR operands
//
// Whether `a` is one this path may serve: exactly torch.Tensor, as the Python probe requires,
// and two-dimensional, as every plan is.
inline bool planned_matmul_torch_csr(const PlannedMatmul& s, PyObject* a) {
  if (reinterpret_cast<PyObject*>(Py_TYPE(a)) != s.tensor_type) return false;
  const at::Tensor& t = THPVariable_Unpack(a);
  return t.layout() == c10::kSparseCsr && t.dim() == 2;
}

// The state dict of `a`'s structure (borrowed), recording `a` as a write witness of its index
// arrays; nullptr with no error set when it has none (torch_structures.h).
inline PyObject* planned_matmul_torch_state(PyObject* a) {
  return torch_structure_lookup(THPVariable_Unpack(a));
}

// Records the first sighting of the product `key` for `a`: plan.install's rule that a plan is
// built only for an operand seen before, applied here so that an operand made afresh for every
// call -- one that is only ever sighted, never planned -- pays for it in C rather than in a
// cold Python function at the end of its ordinary path (2 us of a mkt1 call). Never fails the
// call it follows: any error is cleared.
inline void planned_matmul_record_sighting(const PlannedMatmul& s, PyObject* a, PyObject* key,
                                           const at::Tensor& bt) {
  // A structure Python keeps nothing for has its sightings kept as plain data in its entry
  // (torch_structure_sight): an operand built over new arrays for every call only ever makes first
  // sightings, and each would otherwise keep a dict and a set alive for as long as its arrays.
  TorchSighting sighting;
  if (!planned_matmul_sighting(s, bt, sighting)) return;
  PyObject* state =
      torch_structure_sight(THPVariable_Unpack(a), sighting, 2 * (size_t)s.max_plans);
  if (state == nullptr) return;
  PyObject* seen = PyDict_GetItemWithError(state, s.seen_attr);  // borrowed
  if (seen != nullptr) {
    if (PySet_Check(seen) && PySet_GET_SIZE(seen) < 2 * s.max_plans && PySet_Add(seen, key) < 0)
      PyErr_Clear();
    return;
  }
  if (PyErr_Occurred()) {
    PyErr_Clear();
    return;
  }
  PyObject* fresh = PySet_New(nullptr);
  if (fresh == nullptr || PySet_Add(fresh, key) < 0 ||
      PyDict_SetItem(state, s.seen_attr, fresh) < 0)
    PyErr_Clear();
  Py_XDECREF(fresh);
}

// Records in torch CSR `owner`'s state that its transposed product under `key` is served by the
// carrier plan `plan_obj` (`plan`) with this permutation, unless that is what it already holds.
// `buffer` is the carrier's values buffer: the record gathers into one of its own, of the same
// dtype and length (see TransposedProductPlan). Bounded like its plans
// (plan.MAX_PLANS_PER_OPERAND keys). Never fails the call it follows: any error is cleared. The
// record lives in the state of the operand's structure, made here if it has none, with the weak
// references that drop it with the arrays: the record holds a buffer of nnz values.
inline void planned_matmul_record_transposed(const PlannedMatmul& s, py::handle owner,
                                             PyObject* key, PyObject* plan_obj,
                                             SpmmCsrPlan* plan, const torch::Tensor& buffer,
                                             const torch::Tensor& perm) {
  if (s.tplans_attr == nullptr || plan_obj == nullptr ||
      !planned_matmul_torch_csr(s, owner.ptr())) {
    PyErr_Clear();
    return;
  }
  const at::Tensor& operand = THPVariable_Unpack(owner.ptr());
  PyObject* state = torch_structure_create(operand);
  if (state == nullptr) {
    PyErr_Clear();
    return;
  }
  PyObject* held = PyDict_GetItemWithError(state, s.tplans_attr);  // borrowed
  if (held == nullptr) {
    if (PyErr_Occurred()) {
      PyErr_Clear();
      return;
    }
    py::object fresh = py::reinterpret_steal<py::object>(PyDict_New());
    if (!fresh || PyDict_SetItem(state, s.tplans_attr, fresh.ptr()) < 0) {
      PyErr_Clear();
      return;
    }
    held = fresh.ptr();  // the state holds it now
  }
  if (!PyDict_CheckExact(held)) return;
  auto* impl = at::sparse_csr::get_sparse_csr_impl(operand);
  const at::Tensor& positions = impl->compressed_indices();
  const at::Tensor& coordinates = impl->plain_indices();
  if (PyDict_Contains(held, key) == 1) {
    TransposedProductPlan* current = planned_matmul_find_transposed(held, key);
    if (current != nullptr && current->records(plan, perm) &&
        current->built_from(positions, coordinates))
      return;
  } else if (PyDict_GET_SIZE(held) >= s.max_plans) {
    PyErr_Clear();
    return;
  }
  PyErr_Clear();
  try {
    // Another record of this transpose (another free dimension) lends its buffer.
    torch::Tensor own;
    PyObject* other_key;
    PyObject* other;
    Py_ssize_t pos = 0;
    while (!own.defined() && PyDict_Next(held, &pos, &other_key, &other)) {
      TransposedProductPlan* tp = planned_matmul_find_transposed(held, other_key);
      if (tp != nullptr) own = tp->buffer_for(perm, buffer.scalar_type());
    }
    if (!own.defined()) {
      // An ordinary tensor even under inference_mode, as autograd.py makes the carrier's.
      c10::InferenceMode not_inference(false);
      own = at::empty_like(buffer, at::MemoryFormat::Contiguous);
    }
    py::object fresh = py::cast(TransposedProductPlan(py::reinterpret_borrow<py::object>(plan_obj),
                                                      plan, std::move(own), perm, positions,
                                                      coordinates));
    if (PyDict_SetItem(held, key, fresh.ptr()) < 0) PyErr_Clear();
  } catch (const std::exception&) {
    PyErr_Clear();
  }
}

// What the ordinary path has to do about plans for a torch operand's call this path did not
// serve -- plan.install's early exits, decided here so that only a call that has work for it
// reaches it.
enum class TorchOperandPlanWork {
  kNone,          // a withdrawn product, or a full cache: plan.install would return at once
  kFirstSighting,  // record the sighting here once the ordinary path has returned
  kInstall,       // seen before, or a plan declined: plan.install builds or accounts for it
};

// A torch sparse CSR operand: serve it from its plan, or hand it to the ordinary path under the
// probed keyword, whose value is the key when plan.install has work to do and None otherwise,
// so the Python function neither probes again nor calls the installer for nothing. A call
// whose ordinary path raises records nothing.
// The product of a torch CSR operand `a` no plan serves, made here when it is exactly the call the
// Python function would make: the drop-in SpMM (`ordinary_spmm`) where the product resolves to it
// at this dtype, B a contiguous 2-D tensor of the operand's dtype, a structure every check clears
// (abi_csr_screen), and at float32 a shape the tiling selector's gate declines. The Python function
// builds an STensor around each operand to make that call -- 312 Python calls, 20 of the 26 us such
// a product took on the M5 at 256 rows. The new reference; nullptr with no error set for a call it
// does not make, which the Python function then makes, reporting whatever is wrong with it.
//
// `a` is recorded as a write witness of its arrays, as the Python function's reading of them
// records it (torch_sparse_index_members): the verdicts and int32 copies kept for the arrays are
// then dropped by a write through `a`'s accessors too.
inline PyObject* planned_matmul_torch_ordinary(const PlannedMatmul& s, const at::Tensor& a,
                                               const at::Tensor& b) {
  const at::ScalarType dtype = b.scalar_type();
  const int which = dtype == at::kFloat ? 0 : dtype == at::kDouble ? 1 : -1;
  if (which < 0 || !s.serve_ordinary[which] || s.ordinary_spmm[which] == nullptr) return nullptr;
  if (which == 0 && s.tiling_candidate == nullptr) return nullptr;
  if (!a.device().is_cpu() || !b.device().is_cpu() || b.layout() != c10::kStrided ||
      b.dim() != 2 || !b.is_contiguous() || b.is_neg() || b.is_conj())
    return nullptr;
  auto* impl = at::sparse_csr::get_sparse_csr_impl(a);
  const at::Tensor& values = impl->values();
  const at::Tensor& positions = impl->compressed_indices();
  const at::Tensor& coordinates = impl->plain_indices();
  if (values.scalar_type() != dtype) return nullptr;
  const int64_t rows = a.size(0), cols = a.size(1), k = b.size(1);
  if (b.size(0) != cols) return nullptr;
  abi_watch_register(positions, a);
  abi_watch_register(coordinates, a);
  if (!abi_csr_screen(positions, coordinates, values, rows, cols, s.screen_grain)) return nullptr;
  if (which == 0) {
    PyObject* argv[3] = {PyLong_FromLongLong(cols), PyLong_FromLongLong(k),
                         PyLong_FromLongLong(values.numel())};
    PyObject* verdict = argv[0] && argv[1] && argv[2]
                            ? PyObject_Vectorcall(s.tiling_candidate, argv, 3, nullptr)
                            : nullptr;
    for (PyObject* o : argv) Py_XDECREF(o);
    const bool declined = verdict == Py_False;
    Py_XDECREF(verdict);
    PyErr_Clear();
    if (!declined) return nullptr;
  }
  torch::Tensor out;
  try {
    out = s.ordinary_spmm[which](positions, coordinates, values, b, rows, cols,
                                 planned_matmul_nthreads(s), s.atparallel);
  } catch (...) {
    return nullptr;  // the Python function makes the same call, and reports
  }
  ++s.ordinary_served;
  return THPVariable_Wrap(std::move(out));
}

// The product of two torch sparse CSR operands, made here when it is exactly the call the Python
// function would make: the prebuilt CSR x CSR kernel at the operands' value dtype (serve_spgemm),
// both on the CPU with one dtype, inner extents that match, and structures the function's own
// check clears (abi_csr_screen, order-free). The result is the torch sparse CSR tensor the Python
// function returns for the same product, from the same function (ops.cpp,
// spgemm_csr_torch_result). Before this, such a product always went to the Python function: two
// STensors built around torch's arrays, the prebuilt registry, the binding and an STensor around
// the result -- 234 Python calls, 87% of a warm call's instructions on a 400-row product (sapling
// 84182). The new reference; nullptr with no error set for a call it does not make, which the
// Python function then makes, reporting whatever is wrong with it.
//
// Both operands are recorded as write witnesses of their arrays, as the Python function's reading
// of them records them (torch_sparse_index_members).
inline PyObject* planned_matmul_torch_spgemm(const PlannedMatmul& s, const at::Tensor& a,
                                             const at::Tensor& b) {
  if (s.ordinary_spgemm == nullptr) return nullptr;
  const at::ScalarType dtype = a.scalar_type();
  const int which = planned_matmul_spgemm_slot(dtype);
  if (which < 0 || !s.serve_spgemm[which] || b.scalar_type() != dtype) return nullptr;
  if (!a.device().is_cpu() || !b.device().is_cpu()) return nullptr;
  const int64_t rows = a.size(0), inner = a.size(1), cols = b.size(1);
  if (b.size(0) != inner) return nullptr;
  auto* a_impl = at::sparse_csr::get_sparse_csr_impl(a);
  auto* b_impl = at::sparse_csr::get_sparse_csr_impl(b);
  const at::Tensor& a_positions = a_impl->compressed_indices();
  const at::Tensor& a_coordinates = a_impl->plain_indices();
  const at::Tensor& a_values = a_impl->values();
  const at::Tensor& b_positions = b_impl->compressed_indices();
  const at::Tensor& b_coordinates = b_impl->plain_indices();
  const at::Tensor& b_values = b_impl->values();
  if (a_values.scalar_type() != dtype || b_values.scalar_type() != dtype) return nullptr;
  abi_watch_register(a_positions, a);
  abi_watch_register(a_coordinates, a);
  abi_watch_register(b_positions, b);
  abi_watch_register(b_coordinates, b);
  // Order-free, as the function's own check is: the kernel does not need a row's coordinates in
  // order, so torch.sparse.mm's own unsorted CSR result is made here too.
  if (!abi_csr_screen(a_positions, a_coordinates, a_values, rows, inner, s.screen_grain,
                      /*require_sorted=*/false) ||
      !abi_csr_screen(b_positions, b_coordinates, b_values, inner, cols, s.screen_grain,
                      /*require_sorted=*/false))
    return nullptr;
  at::Tensor out;
  try {
    out = s.ordinary_spgemm(a_positions, a_coordinates, a_values, b_positions, b_coordinates,
                            b_values, rows, inner, cols, s.spgemm_int64_indices);
  } catch (...) {
    return nullptr;  // the Python function makes the same call, and reports
  }
  ++s.spgemm_served;
  return THPVariable_Wrap(std::move(out));
}

inline PyObject* planned_matmul_torch_call(const PlannedMatmul& s, PyObject* a, PyObject* b,
                                           PyObject* const* args, Py_ssize_t nargsf) {
  const at::Tensor& bt = THPVariable_Unpack(b);
  py::object key = py::reinterpret_steal<py::object>(planned_matmul_key(s, bt));
  if (!key) {
    PyErr_Clear();
    return PyObject_Vectorcall(s.ordinary, args, nargsf, nullptr);
  }
  TorchOperandPlanWork work = TorchOperandPlanWork::kFirstSighting;
  const at::Tensor& at_a = THPVariable_Unpack(a);
  TorchStructureFound found;
  if (torch_structure_find(at_a, found) != nullptr) {
    torch_structure_witness(*found.entry, at_a, *found.positions, *found.coordinates);
    PyObject* state = found.entry->state.ptr();  // none for a structure only ever sighted
    PyObject* plans =
        state != nullptr ? PyDict_GetItemWithError(state, s.plans_attr) : nullptr;  // borrowed
    if (plans != nullptr && PyDict_CheckExact(plans)) {
      PyObject* plan_obj = PyDict_GetItemWithError(plans, key.ptr());  // borrowed
      if (plan_obj != nullptr && plan_obj != Py_None) {
        if (SpmmCsrPlan* plan = planned_matmul_cast(plan_obj)) {
          // The members themselves, not aliases: nothing is allocated, and a member is the
          // caller's own array, sharing its version counter. A write through an accessor's alias
          // moves `a`'s counter instead, which the plan's write witnesses read (plan.h).
          auto* impl = at::sparse_csr::get_sparse_csr_impl(at_a);
          c10::optional<torch::Tensor> served =
              plan->run_csr(*found.positions, *found.coordinates, impl->values(), bt,
                            planned_matmul_nthreads(s), s.atparallel);
          if (served.has_value()) return THPVariable_Wrap(std::move(*served));
        }
      }
    }
    PyErr_Clear();
    // Not served. If that is because the arrays have been written, everything the entry holds is
    // stale and is cleared here, before deciding what plan.install has to do.
    torch_structure_revalidate(found);
    plans = state != nullptr ? PyDict_GetItemWithError(state, s.plans_attr) : nullptr;
    if (plans != nullptr && PyDict_CheckExact(plans)) {
      PyObject* plan_obj = PyDict_GetItemWithError(plans, key.ptr());  // borrowed
      if (plan_obj == Py_None) {
        work = TorchOperandPlanWork::kNone;
      } else if (plan_obj != nullptr) {
        work = TorchOperandPlanWork::kInstall;  // a plan that declined: plan.install accounts
      } else if (PyDict_GET_SIZE(plans) >= s.max_plans) {
        work = TorchOperandPlanWork::kNone;
      }
    }
    PyErr_Clear();
    if (work == TorchOperandPlanWork::kFirstSighting) {
      PyObject* seen =
          state != nullptr ? PyDict_GetItemWithError(state, s.seen_attr) : nullptr;  // borrowed
      TorchSighting sighting;
      if ((seen != nullptr && PySet_Check(seen) && PySet_Contains(seen, key.ptr()) == 1) ||
          (state == nullptr && planned_matmul_sighting(s, bt, sighting) &&
           torch_structure_sighted(*found.entry, sighting)))
        work = TorchOperandPlanWork::kInstall;
      PyErr_Clear();
    }
  }
  if (work != TorchOperandPlanWork::kInstall) {
    if (PyObject* served = planned_matmul_torch_ordinary(s, at_a, bt)) {
      if (work == TorchOperandPlanWork::kFirstSighting)
        planned_matmul_record_sighting(s, a, key.ptr(), bt);
      return served;
    }
  }
  PyObject* argv[3] = {a, b, work == TorchOperandPlanWork::kInstall ? key.ptr() : Py_None};
  PyObject* result = PyObject_Vectorcall(s.ordinary, argv, 2, s.probed_kwnames);
  if (result != nullptr && work == TorchOperandPlanWork::kFirstSighting)
    planned_matmul_record_sighting(s, a, key.ptr(), bt);
  return result;
}

// Whether a gradient is wanted through `t`: grad mode is on and it requires grad, or
// forward-mode AD has given it a tangent (torch supports one forward-AD level, level 0). A
// call with such an input is the Python function's, which records it for autograd
// (scorch/autograd.py); a plan would serve it and the result would carry no gradient.
//
// A tensor without autograd metadata can do neither, and that is every tensor of a call made
// without gradients, so the metadata pointer is asked first. Each of these accessors is a call
// into libc10 (torch 2.13 defines none of them inline), and on a cold call each is a cache miss
// in front of the kernel: asking requires_grad and _fw_grad of a tensor without metadata cost a
// cold planned call 1.0-1.4 us on MKT against the tree before autograd; asking for the pointer
// alone, 0.3-0.7 us.
inline bool planned_matmul_wants_grad(const at::Tensor& t) {
  if (t.unsafeGetTensorImpl()->autograd_meta() == nullptr) return false;
  return (t.requires_grad() && at::GradMode::is_enabled()) || t._fw_grad(/*level=*/0).defined();
}

// ---------------------------------------------------------------------------
// A planned product that autograd records
//
// A training step through a small matrix spent its cold time in front of the kernel, not in it.
// On an M5, one cold forward of a 112 x 112 CSR operand with 72 nonzeros times an 8-column dense
// matrix took 49 us through scorch/autograd.py: the kernel call was 18 us of it, and the bare
// machinery of a Python torch.autograd.Function 23 us -- more than torch's whole cold forward of
// the same product (19 us). So the reverse-mode product of a torch CSR operand that has a plan is
// recorded here, by a C++ autograd Function whose forward is the plan's kernel. Its backward is
// the Python function autograd.planned_mm_backward: the same gradients by the same code as the
// Python path's, recorded in turn when the backward is asked to build a graph. The backward was
// never the cold cost.
//
// It serves only products the Python path computes identically: a contiguous two-dimensional
// dense operand (the Python path lays the product out like a transposed one), no forward-mode
// tangent on either operand (a C++ Function has no forward derivative), no functorch transform.
// Every other product, and every operand without a plan yet, goes to the Python function.
//
// The operand itself is the Function's input, as for autograd._TorchSparseMM, so its gradient is
// torch's: the dense G @ B^T torch.matmul gives a sparse operand, not the stored-entries gradient
// one through its values() would be.
//
// The product, computed by the plan before the Function records it, reaches the forward in a
// struct, not as a tensor argument: torch makes a tensor argument that a Function returns into a
// view of that argument (see the note above autograd.py's _x_at_a), so it would have to be
// detached on every call.
struct PlannedMatmulProduct {
  at::Tensor out;
};

struct PlannedMatmulGrad : public torch::autograd::Function<PlannedMatmulGrad> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& a,
                            const at::Tensor& b, PlannedMatmulProduct product) {
    at::Tensor out = std::move(product.out);
    // Only what a requested gradient reads, as autograd._record saves it: b's gradient reads the
    // operand, the operand's reads b, and b can be a large activation. The operand is saved as
    // itself: torch's values() of it shares its version counter, so a write through any alias of
    // the values between this forward and its backward is caught there, as torch.matmul catches
    // it (the values member's counter is its own, and saving it let the backward read new values
    // silently).
    ctx->save_for_backward({b.requires_grad() ? a : at::Tensor(),
                            a.requires_grad() ? b : at::Tensor()});
    // Never a view: see the note above autograd.py's _x_at_a. The kernel allocates the product as
    // the [rows, k] matrix it is (csrc/spmm.h), so `out` is a tensor of its own; it is detached
    // only if it ever arrives as a view.
    return out.is_view() ? out.detach() : out;
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grads) {
    const bool want_a = ctx->needs_input_grad(0), want_b = ctx->needs_input_grad(1);
    torch::autograd::variable_list out(3);  // a, b, and the product (not a tensor)
    if (!grads[0].defined() || !(want_a || want_b)) return out;
    torch::autograd::variable_list saved = ctx->get_saved_variables();  // checks versions
    pybind11::gil_scoped_acquire gil;
    PyObject* fn = planned_matmul_state().mm_backward;
    TORCH_CHECK(fn != nullptr, "scorch: the planned product's backward is no longer installed");
    // An undefined saved tensor (one no requested gradient reads) wraps as None.
    THPObjectPtr a(THPVariable_Wrap(saved[0]));
    THPObjectPtr b(THPVariable_Wrap(saved[1]));
    THPObjectPtr g(THPVariable_Wrap(grads[0]));
    THPObjectPtr result;
    if (a && b && g)
      result = PyObject_CallFunctionObjArgs(fn, a.get(), b.get(), g.get(),
                                            want_a ? Py_True : Py_False,
                                            want_b ? Py_True : Py_False, nullptr);
    if (!result) {
      // Raised again, as itself, by the Python call that ran the backward.
      python_error err;
      err.persist();
      throw std::move(err);
    }
    TORCH_CHECK(PyTuple_Check(result.get()) && PyTuple_GET_SIZE(result.get()) == 2,
                "scorch: planned_mm_backward must return (a's gradient, b's gradient)");
    for (Py_ssize_t i = 0; i < 2; ++i) {
      PyObject* item = PyTuple_GET_ITEM(result.get(), i);
      if (item == Py_None) continue;
      TORCH_CHECK(THPVariable_Check(item), "scorch: planned_mm_backward returned a non-tensor");
      out[i] = THPVariable_Unpack(item);
    }
    return out;
  }
};

// A torch CSR operand's product that autograd must record, served from its plan: the result as a
// new reference, or nullptr with no error set when the Python function is to compute it instead.
inline PyObject* planned_matmul_torch_grad(const PlannedMatmul& s, PyObject* a, PyObject* b) {
  const at::Tensor& at_a = THPVariable_Unpack(a);
  const at::Tensor& bt = THPVariable_Unpack(b);
  if (s.mm_backward == nullptr || bt.layout() != c10::kStrided || bt.dim() != 2 ||
      !bt.is_contiguous() ||
      at_a._fw_grad(/*level=*/0).defined() || bt._fw_grad(/*level=*/0).defined() ||
      at::functorch::functorchTLSAccessor())
    return nullptr;
  py::object key = py::reinterpret_steal<py::object>(planned_matmul_key(s, bt));
  SpmmCsrPlan* plan = nullptr;
  if (key) {
    if (PyObject* state = planned_matmul_torch_state(a))
      plan = planned_matmul_find(PyDict_GetItemWithError(state, s.plans_attr), key.ptr());
  }
  PyErr_Clear();
  if (plan == nullptr) return nullptr;
  auto* impl = at::sparse_csr::get_sparse_csr_impl(at_a);
  c10::optional<torch::Tensor> served;
  {
    at::AutoGradMode no_grad(false);  // the product is recorded below, once
    served = plan->run_csr(impl->compressed_indices(), impl->plain_indices(), impl->values(), bt,
                           planned_matmul_nthreads(s), s.atparallel);
  }
  if (!served.has_value()) return nullptr;
  return THPVariable_Wrap(
      PlannedMatmulGrad::apply(at_a, bt, PlannedMatmulProduct{std::move(*served)}));
}

// x @ A's recorded product, as autograd._TorchSparseXA records it: A^T @ b with b = x^T, returned
// as x @ A, its transpose, which the caller receives as it is, with the operand itself as the
// input (see PlannedMatmulGrad). Its backward is autograd.planned_mm_t_backward on the gradient's
// transpose (the gradient of A^T @ b), the same gradients by the same code as _TorchSparseXA's.
struct PlannedMatmulGradT : public torch::autograd::Function<PlannedMatmulGradT> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& a,
                            const at::Tensor& b, PlannedMatmulProduct product,
                            bool b_transposed) {
    at::Tensor out = std::move(product.out);
    ctx->saved_data["b_transposed"] = b_transposed;
    // As PlannedMatmulGrad saves them: b's gradient reads the operand, the operand's reads b.
    ctx->save_for_backward({b.requires_grad() ? a : at::Tensor(),
                            a.requires_grad() ? b : at::Tensor()});
    // Never a view: see the note above autograd.py's _x_at_a. planned_matmul_torch_right makes x @ A
    // a tensor of its own (planned_matmul_x_at_a); this detaches it only where it could not.
    return out.is_view() ? out.detach() : out;
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grads) {
    const bool want_a = ctx->needs_input_grad(0), want_b = ctx->needs_input_grad(1);
    torch::autograd::variable_list out(4);  // a, b, the product and the layout flag
    if (!grads[0].defined() || !(want_a || want_b)) return out;
    torch::autograd::variable_list saved = ctx->get_saved_variables();  // checks versions
    pybind11::gil_scoped_acquire gil;
    PyObject* fn = planned_matmul_state().mm_t_backward;
    TORCH_CHECK(fn != nullptr, "scorch: x @ A's planned backward is no longer installed");
    THPObjectPtr a(THPVariable_Wrap(saved[0]));
    THPObjectPtr b(THPVariable_Wrap(saved[1]));
    // the gradient of x @ A, transposed: that of A^T @ b, as _TorchSparseXA.backward passes it
    THPObjectPtr g(THPVariable_Wrap(grads[0].t()));
    const bool b_transposed = ctx->saved_data["b_transposed"].toBool();
    THPObjectPtr result;
    if (a && b && g)
      result = PyObject_CallFunctionObjArgs(fn, a.get(), b.get(), g.get(),
                                            want_a ? Py_True : Py_False,
                                            want_b ? Py_True : Py_False,
                                            b_transposed ? Py_True : Py_False, nullptr);
    if (!result) {
      python_error err;
      err.persist();
      throw std::move(err);
    }
    TORCH_CHECK(PyTuple_Check(result.get()) && PyTuple_GET_SIZE(result.get()) == 2,
                "scorch: planned_mm_t_backward must return (a's gradient, b's gradient)");
    for (Py_ssize_t i = 0; i < 2; ++i) {
      PyObject* item = PyTuple_GET_ITEM(result.get(), i);
      if (item == Py_None) continue;
      TORCH_CHECK(THPVariable_Check(item), "scorch: planned_mm_t_backward returned a non-tensor");
      out[i] = THPVariable_Unpack(item);
    }
    return out;
  }
};

// t's transpose, contiguous, as autograd._transpose makes it: scorch's blocked transpose for
// float32 (ops.fast_transpose, on the same thread count), torch's copy otherwise.
inline at::Tensor planned_matmul_dense_transpose(const PlannedMatmul& s, const at::Tensor& t) {
  if (t.scalar_type() == at::kFloat)
    return scorch_transpose_2d_float(t, static_cast<int>(planned_matmul_nthreads(s)));
  return t.t().contiguous();
}

// x @ A from the plan's product A^T @ b (b = x^T), as autograd._x_at_a makes it, and a tensor of
// its own: for a transposed b, the product's transposed copy (autograd._in_layout's, before it is
// restrided back), which is x @ A contiguous; otherwise (x one row or column) the product's
// transpose, restrided in place when nothing else holds the product (the plan's fresh buffer).
// Neither adds a copy; only a transpose that arrives as a view is detached.
inline at::Tensor planned_matmul_x_at_a(const PlannedMatmul& s, at::Tensor product,
                                        bool b_transposed) {
  if (b_transposed) {
    at::Tensor t = planned_matmul_dense_transpose(s, product);
    return t.is_view() ? t.detach() : t;
  }
  if (product.use_count() == 1 && !product.is_view() &&
      product.unsafeGetTensorImpl()->allow_tensor_metadata_change()) {
    product.t_();
    return product;
  }
  return product.t().detach();  // a new handle, not a view
}

// x @ A for a dense `x` and a torch CSR `a` whose transposed product holds a TransposedProductPlan
// (see above), computed as autograd.py computes it and, when `grad`, recorded as it records it:
// _TorchSparseXA.apply(op, a, x.t()), or right_product's unrecorded (A^T @ x^T)^T, each x @ A
// itself and not a view (planned_matmul_x_at_a). The result
// as a new reference, or nullptr with no error set when the Python function is to compute it.
//
// Served only for a contiguous float32 or float64 CPU x of A's dtype whose columns are A's rows,
// exactly torch.Tensor (a Parameter goes to the Python function, as for A @ B), with no
// forward-mode tangent and no functorch transform; anything else, including a shape error to raise,
// goes there.
inline PyObject* planned_matmul_torch_right(const PlannedMatmul& s, PyObject* x, PyObject* a,
                                            bool grad) {
  if (s.tplans_attr == nullptr || (grad && s.mm_t_backward == nullptr)) return nullptr;
  const at::Tensor& xt = THPVariable_Unpack(x);
  const at::Tensor& at_a = THPVariable_Unpack(a);
  const at::ScalarType dtype = xt.scalar_type();
  if (xt.layout() != c10::kStrided || xt.dim() != 2 || !xt.is_contiguous() || xt.is_neg() ||
      xt.is_conj() || !xt.device().is_cpu() || !at_a.device().is_cpu() ||
      (dtype != at::kFloat && dtype != at::kDouble) || at_a.scalar_type() != dtype ||
      xt.size(1) != at_a.size(0))
    return nullptr;
  if (at::functorch::functorchTLSAccessor() ||
      (grad && (xt._fw_grad(/*level=*/0).defined() || at_a._fw_grad(/*level=*/0).defined())))
    return nullptr;
  // b = x^T, the dense operand of A^T @ b, keyed by its shape as the carrier's plan is.
  const int64_t sizes[2] = {xt.size(1), xt.size(0)};
  py::object key = py::reinterpret_steal<py::object>(planned_matmul_key_sizes(s, sizes, dtype));
  TransposedProductPlan* tp = nullptr;
  PyObject* held = nullptr;  // borrowed from the operand's state
  if (key) {
    if (PyObject* state = planned_matmul_torch_state(a)) {
      held = PyDict_GetItemWithError(state, s.tplans_attr);
      tp = planned_matmul_find_transposed(held, key.ptr());
    }
  }
  PyErr_Clear();
  if (tp == nullptr) return nullptr;
  auto* impl = at::sparse_csr::get_sparse_csr_impl(at_a);
  if (!tp->built_from(impl->compressed_indices(), impl->plain_indices())) {
    // Recorded before a write to the operand: it can never serve again, and it holds the plan of a
    // transpose autograd.py is about to rebuild. The product that next serves records afresh.
    if (PyDict_DelItem(held, key.ptr()) < 0) PyErr_Clear();
    return nullptr;
  }
  // autograd._transposed_layout(b): b = x^T is the transpose of a contiguous matrix unless it is
  // contiguous itself, which it is when x has one row or one column.
  at::Tensor b = xt.t();
  const bool b_transposed = !b.is_contiguous();
  at::Tensor out;  // x @ A
  {
    at::AutoGradMode no_grad(false);  // recorded below, once
    c10::optional<torch::Tensor> served;
    {
      // autograd._plain(b): b's transpose's transpose, made contiguous by scorch's transpose.
      // Released before the result's transpose is allocated, as the Python function releases it.
      at::Tensor m = b_transposed ? planned_matmul_dense_transpose(s, xt) : b;
      served = tp->run(impl->values().contiguous(), m, planned_matmul_nthreads(s), s.atparallel);
    }
    if (!served.has_value()) return nullptr;
    out = planned_matmul_x_at_a(s, std::move(*served), b_transposed);
  }
  if (!grad) return THPVariable_Wrap(std::move(out));
  return THPVariable_Wrap(
      PlannedMatmulGradT::apply(at_a, b, PlannedMatmulProduct{std::move(out)}, b_transposed));
}

inline PyObject* planned_matmul_call(PyObject* /*module*/, PyObject* const* args,
                                     Py_ssize_t nargsf, PyObject* kwnames) {
  HANDLE_TH_ERRORS
  const PlannedMatmul& s = planned_matmul_state();
  // The forced schedule is read last, so only a call this entry could serve pays for it; a
  // forced call goes to the Python function untouched, with no plan probed or sighted.
  if (PyVectorcall_NARGS(nargsf) == 2 && kwnames == nullptr &&
      reinterpret_cast<PyObject*>(Py_TYPE(args[1])) == s.tensor_type &&
      PyList_CheckExact(s.enabled) && PyList_GET_SIZE(s.enabled) >= 1 &&
      PyList_GET_ITEM(s.enabled, 0) == Py_True && !planned_matmul_schedule_forced(s)) {
    const bool b_grad = planned_matmul_wants_grad(THPVariable_Unpack(args[1]));
    if (reinterpret_cast<PyObject*>(Py_TYPE(args[0])) == s.stensor_type) {
      if (!b_grad) {
        PyObject* served = planned_matmul_try(s, args[0], args[1]);
        if (served != nullptr || PyErr_Occurred()) return served;
      }
    } else if (planned_matmul_torch_csr(s, args[0])) {
      const bool a_grad = planned_matmul_wants_grad(THPVariable_Unpack(args[0]));
      if (planned_matmul_torch_csr(s, args[1])) {
        // Two torch CSR operands: a CSR x CSR product, which no CSR x dense plan serves. A call
        // that wants a gradient is autograd.py's, in the Python function; any other is made here
        // or handed over as probed, which is what the CSR x dense path above hands such a call
        // over as when it has no plan for it.
        if (b_grad || a_grad) return PyObject_Vectorcall(s.ordinary, args, nargsf, kwnames);
        PyObject* served = planned_matmul_torch_spgemm(s, THPVariable_Unpack(args[0]),
                                                       THPVariable_Unpack(args[1]));
        if (served != nullptr || PyErr_Occurred()) return served;
        PyObject* argv[3] = {args[0], args[1], Py_None};
        return PyObject_Vectorcall(s.ordinary, argv, 2, s.probed_kwnames);
      }
      if (!b_grad && !a_grad) return planned_matmul_torch_call(s, args[0], args[1], args, nargsf);
      PyObject* served = planned_matmul_torch_grad(s, args[0], args[1]);
      if (served != nullptr || PyErr_Occurred()) return served;
    } else if (reinterpret_cast<PyObject*>(Py_TYPE(args[0])) == s.tensor_type &&
               planned_matmul_torch_csr(s, args[1])) {
      const bool grad = b_grad || planned_matmul_wants_grad(THPVariable_Unpack(args[0]));
      PyObject* served = planned_matmul_torch_right(s, args[0], args[1], grad);
      if (served != nullptr || PyErr_Occurred()) return served;
    }
  }
  return PyObject_Vectorcall(s.ordinary, args, nargsf, kwnames);
  END_HANDLE_TH_ERRORS
}

inline void planned_matmul_set(PyObject*& slot, PyObject* value) {
  Py_XINCREF(value);
  Py_XSETREF(slot, value);
}

inline py::object install_planned_matmul(py::object ordinary, py::object stensor_type,
                                         py::object tensor_type, py::object generation,
                                         py::object enabled, const std::string& plans_attr,
                                         const std::string& seen_attr,
                                         Py_ssize_t max_plans, const std::string& probed_kw,
                                         bool match_host_threads, bool atparallel,
                                         py::object module, const std::string& doc) {
  if (!PyCallable_Check(ordinary.ptr())) throw py::type_error("ordinary must be callable");
  if (!PyType_Check(stensor_type.ptr()) || !PyType_Check(tensor_type.ptr()))
    throw py::type_error("stensor_type and tensor_type must be types");
  if (!PyList_CheckExact(generation.ptr()) || PyList_GET_SIZE(generation.ptr()) < 1 ||
      !PyList_CheckExact(enabled.ptr()) || PyList_GET_SIZE(enabled.ptr()) < 1)
    throw py::type_error("generation and enabled must be one-element lists");
  PlannedMatmul& s = planned_matmul_state();
  planned_matmul_set(s.ordinary, ordinary.ptr());
  planned_matmul_set(s.stensor_type, stensor_type.ptr());
  planned_matmul_set(s.tensor_type, tensor_type.ptr());
  planned_matmul_set(s.generation, generation.ptr());
  planned_matmul_set(s.enabled, enabled.ptr());
  Py_XSETREF(s.plans_attr, PyUnicode_InternFromString(plans_attr.c_str()));
  Py_XSETREF(s.seen_attr, PyUnicode_InternFromString(seen_attr.c_str()));
  Py_XSETREF(s.probed_kwnames, Py_BuildValue("(s)", probed_kw.c_str()));
  if (s.storage_attr == nullptr) s.storage_attr = PyUnicode_InternFromString("_storage");
  if (s.value_attr == nullptr) s.value_attr = PyUnicode_InternFromString("_value");
  if (s.metadata_attr == nullptr) s.metadata_attr = PyUnicode_InternFromString("_metadata");
  if (s.requires_grad_attr == nullptr)
    s.requires_grad_attr = PyUnicode_InternFromString("requires_grad");
  if (s.plans_attr == nullptr || s.seen_attr == nullptr || s.probed_kwnames == nullptr ||
      s.storage_attr == nullptr || s.value_attr == nullptr || s.metadata_attr == nullptr ||
      s.requires_grad_attr == nullptr)
    throw py::error_already_set();
  torch_structures().seen_attr = py::reinterpret_borrow<py::object>(s.seen_attr);
  s.max_plans = max_plans;
  s.match_host_threads = match_host_threads;
  s.atparallel = atparallel;

  // The method table outlives every function object made from it. The docstring is kept from
  // the first install: a function object holds a pointer into it.
  static std::string doc_text = "matmul(a, b, /, **kwargs)\n--\n\n" + doc;
  static PyMethodDef def = {
      "matmul",
      reinterpret_cast<PyCFunction>(reinterpret_cast<void (*)(void)>(planned_matmul_call)),
      METH_FASTCALL | METH_KEYWORDS, doc_text.c_str()};
  // `self` is the module, so the function reports __qualname__ "matmul" and pickles by
  // reference as module.matmul, exactly as the Python function did.
  PyObject* name = PyObject_GetAttrString(module.ptr(), "__name__");
  if (name == nullptr) throw py::error_already_set();
  PyObject* fn = PyCFunction_NewEx(&def, module.ptr(), name);
  Py_DECREF(name);
  if (fn == nullptr) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(fn);
}

// Lets the C entry point make a torch CSR operand's unplanned product itself (see
// planned_matmul_torch_ordinary): `tiling_candidate` is tiling.shape_is_candidate (None withdraws
// float32), `float32`/`float64` whether the CSR x dense product resolves to the drop-in SpMM at
// that dtype, `screen_grain` storage._WRAP_GRAIN.
inline void set_planned_matmul_ordinary(py::object tiling_candidate, bool float32, bool float64,
                                        int64_t screen_grain) {
  if (!tiling_candidate.is_none() && !PyCallable_Check(tiling_candidate.ptr()))
    throw py::type_error("tiling_candidate must be callable or None");
  if (screen_grain < 1) throw py::value_error("screen_grain must be positive");
  PlannedMatmul& s = planned_matmul_state();
  planned_matmul_set(s.tiling_candidate,
                     tiling_candidate.is_none() ? nullptr : tiling_candidate.ptr());
  s.serve_ordinary[0] = float32;
  s.serve_ordinary[1] = float64;
  s.screen_grain = screen_grain;
}

}  // namespace scorch_native
