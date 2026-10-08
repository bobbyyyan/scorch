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
// sparse CSR layout), whose plans plan.py keeps in plan.TORCH_OPERANDS, keyed on id(a) and
// checked against a weak reference, because a torch tensor cannot carry them (see there).
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

#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/FuncTorchTLS.h>
#include <ATen/core/grad_mode.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/object_ptr.h>

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
  // Torch sparse CSR operands: their plans and sightings live in plan.TORCH_OPERANDS.
  PyObject* torch_operands = nullptr;    // plan.TORCH_OPERANDS: id -> (weak reference, state)
  PyObject* operand_ref_type = nullptr;  // plan._OperandRef, the weak reference an entry holds
  PyObject* operand_died = nullptr;      // plan._drop_torch_operand, its callback
  PyObject* seen_attr = nullptr;         // interned plan.SEEN_ATTR
  PyObject* key_attr = nullptr;          // interned "key", _OperandRef's slot
  PyObject* probed_kwnames = nullptr;    // (ops._C_PROBED_KW,): the keyword the key travels under
  Py_ssize_t max_plans = 0;              // plan.MAX_PLANS_PER_OPERAND
  bool match_host_threads = false;       // ops._MATCH_HOST_THREADS
  bool atparallel = false;               // ops._PLAN_ATPARALLEL
  // autograd.planned_mm_backward: the backward of a product PlannedMatmulGrad recorded. Unset,
  // every product autograd must record goes to the Python function.
  PyObject* mm_backward = nullptr;
  // x @ A for a torch CSR A (see planned_matmul_torch_right): autograd.planned_mm_t_backward, the
  // backward of a product PlannedMatmulGradT recorded, and the interned name under which an
  // operand's state holds its TransposedProductPlans (plan.TRANSPOSED_PLANS_ATTR). Unset, every
  // x @ A goes to the Python function.
  PyObject* mm_t_backward = nullptr;
  PyObject* tplans_attr = nullptr;
};

inline PlannedMatmul& planned_matmul_state() {
  static PlannedMatmul state;
  return state;
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
// through the operand's own state: whenever planned_matmul_gathered serves the transposed product
// of a torch CSR operand (its `owner`), it records in that operand's state a TransposedProductPlan
// for it -- the carrier's plan, the permutation, and the operand's index arrays the transpose was
// built from -- under the key that product's plan is stored under. A call is served only while
// the operand is as it was then: the same index buffers and lengths, and the same version. The
// version is the operand's own counter, which torch shares with every alias its crow_indices(),
// col_indices() and values() return, so a write through any of them moves it, while the member
// arrays read here keep counters of their own that such a write does not move. It is the counter
// autograd.py's structure cache compares (through the aliases), so a call this declines is one
// whose structure and transpose that cache rebuilds.
//
// The transposed values are gathered into a buffer of the record's own, not the carrier's. The
// carrier's buffer belongs to autograd.py's structure, shared by every operand over the same index
// arrays, and its writers serialize on that structure's lock, which they hold across gather_values
// with the GIL released; a writer here holds only the GIL. So the record's buffer is written only
// here, with the GIL held from the fill to the end of the product (plan runs do not release it),
// and an operand's records of one transpose share one buffer.
struct TransposedProductPlan {
  TransposedProductPlan(py::object plan, SpmmCsrPlan* product, torch::Tensor buffer,
                        torch::Tensor perm, torch::Tensor positions, torch::Tensor coordinates,
                        uint32_t operand_version)
      : plan_(std::move(plan)),
        product_(product),
        buffer_(std::move(buffer)),
        perm_(std::move(perm)),
        positions_(std::move(positions)),
        coordinates_(std::move(coordinates)),
        positions_data_(positions_.data_ptr()),
        coordinates_data_(coordinates_.data_ptr()),
        operand_version_(operand_version) {}

  // Whether an operand whose members are these index arrays, at this version, is the one this
  // was recorded from, unchanged since. It holds the arrays it was recorded from, so no other
  // buffer can occupy the recorded addresses.
  bool built_from(const at::Tensor& positions, const at::Tensor& coordinates,
                  uint32_t operand_version) const {
    return positions.defined() && coordinates.defined() &&
           positions.data_ptr() == positions_data_ &&
           coordinates.data_ptr() == coordinates_data_ &&
           positions.numel() == positions_.numel() &&
           coordinates.numel() == coordinates_.numel() && operand_version == operand_version_;
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
  torch::Tensor positions_;
  torch::Tensor coordinates_;
  const void* positions_data_;
  const void* coordinates_data_;
  uint32_t operand_version_;
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

// The state dict of `a`'s entry in plan.TORCH_OPERANDS (borrowed), or nullptr with no error set
// when it has none -- including an entry left by an earlier object with the same id, whose weak
// reference no longer names `a`.
inline PyObject* planned_matmul_torch_state(const PlannedMatmul& s, PyObject* a, PyObject* id) {
  if (s.torch_operands == nullptr || !PyDict_CheckExact(s.torch_operands)) return nullptr;
  PyObject* held = PyDict_GetItemWithError(s.torch_operands, id);  // borrowed
  if (held == nullptr || !PyTuple_CheckExact(held) || PyTuple_GET_SIZE(held) != 2) {
    PyErr_Clear();
    return nullptr;
  }
  PyObject* ref = PyTuple_GET_ITEM(held, 0);
  if (!PyWeakref_Check(ref) || PyWeakref_GetObject(ref) != a) return nullptr;
  PyObject* state = PyTuple_GET_ITEM(held, 1);
  return PyDict_CheckExact(state) ? state : nullptr;
}

// Records the first sighting of the product `key` for `a`: plan.install's rule that a plan is
// built only for an operand seen before, applied here so that an operand made afresh for every
// call -- one that is only ever sighted, never planned -- pays for it in C rather than in a
// cold Python function at the end of its ordinary path (2 us of a mkt1 call). Never fails the
// call it follows: any error is cleared.
inline void planned_matmul_record_sighting(const PlannedMatmul& s, PyObject* a, PyObject* id,
                                           PyObject* key) {
  PyObject* state = planned_matmul_torch_state(s, a, id);
  if (state != nullptr) {
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
    return;
  }
  // No entry yet: (weak reference carrying its key, {SEEN_ATTR: {key}}), as
  // plan.torch_operand_state(a, create=True) would make it.
  PyObject* ref = PyObject_CallFunctionObjArgs(s.operand_ref_type, a, s.operand_died, nullptr);
  if (ref == nullptr) {
    PyErr_Clear();
    return;
  }
  PyObject* seen = PySet_New(nullptr);
  PyObject* fresh_state = PyDict_New();
  PyObject* entry = nullptr;
  bool ok = seen != nullptr && fresh_state != nullptr &&
            PyObject_SetAttr(ref, s.key_attr, id) == 0 && PySet_Add(seen, key) == 0 &&
            PyDict_SetItem(fresh_state, s.seen_attr, seen) == 0;
  if (ok) {
    entry = PyTuple_Pack(2, ref, fresh_state);
    ok = entry != nullptr && PyDict_SetItem(s.torch_operands, id, entry) == 0;
  }
  if (!ok) PyErr_Clear();
  Py_XDECREF(entry);
  Py_XDECREF(fresh_state);
  Py_XDECREF(seen);
  Py_DECREF(ref);
}

// Records in torch CSR `owner`'s state that its transposed product under `key` is served by the
// carrier plan `plan_obj` (`plan`) with this permutation, unless that is what it already holds.
// `buffer` is the carrier's values buffer: the record gathers into one of its own, of the same
// dtype and length (see TransposedProductPlan). Bounded like its plans
// (plan.MAX_PLANS_PER_OPERAND keys). Never fails the call it follows: any error is cleared. Only
// an operand with an entry in plan.TORCH_OPERANDS has a state to record in, and autograd.py makes
// that entry before it computes the product.
inline void planned_matmul_record_transposed(const PlannedMatmul& s, py::handle owner,
                                             PyObject* key, PyObject* plan_obj,
                                             SpmmCsrPlan* plan, const torch::Tensor& buffer,
                                             const torch::Tensor& perm) {
  if (s.tplans_attr == nullptr || plan_obj == nullptr ||
      !planned_matmul_torch_csr(s, owner.ptr())) {
    PyErr_Clear();
    return;
  }
  py::object id = py::reinterpret_steal<py::object>(PyLong_FromVoidPtr(owner.ptr()));
  PyObject* state = id ? planned_matmul_torch_state(s, owner.ptr(), id.ptr()) : nullptr;
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
  const at::Tensor& operand = THPVariable_Unpack(owner.ptr());
  auto* impl = at::sparse_csr::get_sparse_csr_impl(operand);
  const at::Tensor& positions = impl->compressed_indices();
  const at::Tensor& coordinates = impl->plain_indices();
  const uint32_t version = abi_version_of(operand);
  if (PyDict_Contains(held, key) == 1) {
    TransposedProductPlan* current = planned_matmul_find_transposed(held, key);
    if (current != nullptr && current->records(plan, perm) &&
        current->built_from(positions, coordinates, version))
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
                                                      coordinates, version));
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
inline PyObject* planned_matmul_torch_call(const PlannedMatmul& s, PyObject* a, PyObject* b,
                                           PyObject* const* args, Py_ssize_t nargsf) {
  const at::Tensor& bt = THPVariable_Unpack(b);
  py::object key = py::reinterpret_steal<py::object>(planned_matmul_key(s, bt));
  py::object id = py::reinterpret_steal<py::object>(key ? PyLong_FromVoidPtr(a) : nullptr);
  if (!key || !id) {
    PyErr_Clear();
    return PyObject_Vectorcall(s.ordinary, args, nargsf, nullptr);
  }
  TorchOperandPlanWork work = TorchOperandPlanWork::kFirstSighting;
  if (PyObject* state = planned_matmul_torch_state(s, a, id.ptr())) {
    PyObject* plans = PyDict_GetItemWithError(state, s.plans_attr);  // borrowed
    PyObject* plan_obj = nullptr;
    if (plans != nullptr && PyDict_CheckExact(plans)) {
      plan_obj = PyDict_GetItemWithError(plans, key.ptr());  // borrowed
      if (plan_obj == Py_None) {
        work = TorchOperandPlanWork::kNone;
      } else if (plan_obj != nullptr) {
        work = TorchOperandPlanWork::kInstall;
        if (SpmmCsrPlan* plan = planned_matmul_cast(plan_obj)) {
          // The members themselves, not aliases: nothing is allocated, and a member shares
          // its version counter with every alias the caller may have written through.
          auto* impl = at::sparse_csr::get_sparse_csr_impl(THPVariable_Unpack(a));
          c10::optional<torch::Tensor> served =
              plan->run_csr(impl->compressed_indices(), impl->plain_indices(), impl->values(),
                            bt, planned_matmul_nthreads(s), s.atparallel);
          if (served.has_value()) return THPVariable_Wrap(std::move(*served));
        }
      } else if (PyDict_GET_SIZE(plans) >= s.max_plans) {
        work = TorchOperandPlanWork::kNone;
      }
    }
    PyErr_Clear();
    if (work == TorchOperandPlanWork::kFirstSighting) {
      PyObject* seen = PyDict_GetItemWithError(state, s.seen_attr);  // borrowed
      if (seen != nullptr && PySet_Check(seen) && PySet_Contains(seen, key.ptr()) == 1)
        work = TorchOperandPlanWork::kInstall;
      PyErr_Clear();
    }
  }
  PyObject* argv[3] = {a, b, work == TorchOperandPlanWork::kInstall ? key.ptr() : Py_None};
  PyObject* result = PyObject_Vectorcall(s.ordinary, argv, 2, s.probed_kwnames);
  if (result != nullptr && work == TorchOperandPlanWork::kFirstSighting)
    planned_matmul_record_sighting(s, a, id.ptr(), key.ptr());
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
struct PlannedMatmulProduct {
  at::Tensor a;    // the operand: the backward finds the structure its gradients run on
  at::Tensor out;  // the product, computed by the plan before the Function records it
};

struct PlannedMatmulGrad : public torch::autograd::Function<PlannedMatmulGrad> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& values,
                            const at::Tensor& b, PlannedMatmulProduct product) {
    ctx->saved_data["a"] = std::move(product.a);
    // Only what a requested gradient reads, as autograd._record saves it: b's gradient reads the
    // values, the values' gradient reads b, and b can be a large activation.
    ctx->save_for_backward({b.requires_grad() ? values : at::Tensor(),
                            values.requires_grad() ? b : at::Tensor()});
    return std::move(product.out);
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grads) {
    const bool want_values = ctx->needs_input_grad(0), want_b = ctx->needs_input_grad(1);
    torch::autograd::variable_list out(3);  // values, b, and the product (not a tensor)
    if (!grads[0].defined() || !(want_values || want_b)) return out;
    torch::autograd::variable_list saved = ctx->get_saved_variables();  // checks versions
    pybind11::gil_scoped_acquire gil;
    PyObject* fn = planned_matmul_state().mm_backward;
    TORCH_CHECK(fn != nullptr, "scorch: the planned product's backward is no longer installed");
    THPObjectPtr a(THPVariable_Wrap(ctx->saved_data["a"].toTensor()));
    THPObjectPtr values(THPVariable_Wrap(saved[0]));
    THPObjectPtr b(THPVariable_Wrap(saved[1]));
    THPObjectPtr g(THPVariable_Wrap(grads[0]));
    THPObjectPtr result;
    if (a && values && b && g)
      result = PyObject_CallFunctionObjArgs(fn, a.get(), values.get(), b.get(), g.get(),
                                            want_values ? Py_True : Py_False,
                                            want_b ? Py_True : Py_False, nullptr);
    if (!result) {
      // Raised again, as itself, by the Python call that ran the backward.
      python_error err;
      err.persist();
      throw std::move(err);
    }
    TORCH_CHECK(PyTuple_Check(result.get()) && PyTuple_GET_SIZE(result.get()) == 2,
                "scorch: planned_mm_backward must return (values' gradient, b's gradient)");
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
  py::object id = py::reinterpret_steal<py::object>(key ? PyLong_FromVoidPtr(a) : nullptr);
  SpmmCsrPlan* plan = nullptr;
  if (key && id) {
    if (PyObject* state = planned_matmul_torch_state(s, a, id.ptr()))
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
  // torch's values(), not the member the plan read: it is a view of `a` that shares `a`'s version
  // counter, so a write through any alias of the values between this forward and its backward is
  // caught there, as torch.sparse.mm and the Python path catch it (the member's counter is its
  // own, and saving it let the backward read the new values silently); and when `a` requires grad
  // the values' gradient reaches `a` by torch's own values() backward.
  at::Tensor values = at_a.values();
  return THPVariable_Wrap(
      PlannedMatmulGrad::apply(values, bt, PlannedMatmulProduct{at_a, std::move(*served)}));
}

// x @ A's recorded product, as autograd._SparseMMT records it: A(values)^T @ b with b = x^T, whose
// output the caller receives transposed. Its backward is autograd.planned_mm_t_backward, the same
// gradients by the same code as _SparseMMT's.
struct PlannedMatmulGradT : public torch::autograd::Function<PlannedMatmulGradT> {
  static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& values,
                            const at::Tensor& b, PlannedMatmulProduct product, bool b_transposed) {
    ctx->saved_data["a"] = std::move(product.a);
    ctx->saved_data["b_transposed"] = b_transposed;
    // As autograd._record saves them: b's gradient reads the values, the values' gradient reads b.
    ctx->save_for_backward({b.requires_grad() ? values : at::Tensor(),
                            values.requires_grad() ? b : at::Tensor()});
    return std::move(product.out);
  }

  static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx,
                                                 torch::autograd::variable_list grads) {
    const bool want_values = ctx->needs_input_grad(0), want_b = ctx->needs_input_grad(1);
    torch::autograd::variable_list out(4);  // values, b, the product and the layout flag
    if (!grads[0].defined() || !(want_values || want_b)) return out;
    torch::autograd::variable_list saved = ctx->get_saved_variables();  // checks versions
    pybind11::gil_scoped_acquire gil;
    PyObject* fn = planned_matmul_state().mm_t_backward;
    TORCH_CHECK(fn != nullptr, "scorch: x @ A's planned backward is no longer installed");
    THPObjectPtr a(THPVariable_Wrap(ctx->saved_data["a"].toTensor()));
    THPObjectPtr values(THPVariable_Wrap(saved[0]));
    THPObjectPtr b(THPVariable_Wrap(saved[1]));
    THPObjectPtr g(THPVariable_Wrap(grads[0]));
    const bool b_transposed = ctx->saved_data["b_transposed"].toBool();
    THPObjectPtr result;
    if (a && values && b && g)
      result = PyObject_CallFunctionObjArgs(fn, a.get(), values.get(), b.get(), g.get(),
                                            want_values ? Py_True : Py_False,
                                            want_b ? Py_True : Py_False,
                                            b_transposed ? Py_True : Py_False, nullptr);
    if (!result) {
      python_error err;
      err.persist();
      throw std::move(err);
    }
    TORCH_CHECK(PyTuple_Check(result.get()) && PyTuple_GET_SIZE(result.get()) == 2,
                "scorch: planned_mm_t_backward must return (values' gradient, b's gradient)");
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

// x @ A for a dense `x` and a torch CSR `a` whose transposed product holds a TransposedProductPlan
// (see above), computed as autograd.py computes it and, when `grad`, recorded as it records it:
// _SparseMMT.apply(op, values, x.t()).t(), or right_product's unrecorded (A^T @ x^T)^T. The result
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
  py::object id = py::reinterpret_steal<py::object>(key ? PyLong_FromVoidPtr(a) : nullptr);
  TransposedProductPlan* tp = nullptr;
  PyObject* held = nullptr;  // borrowed from the operand's state
  if (key && id) {
    if (PyObject* state = planned_matmul_torch_state(s, a, id.ptr())) {
      held = PyDict_GetItemWithError(state, s.tplans_attr);
      tp = planned_matmul_find_transposed(held, key.ptr());
    }
  }
  PyErr_Clear();
  if (tp == nullptr) return nullptr;
  auto* impl = at::sparse_csr::get_sparse_csr_impl(at_a);
  if (!tp->built_from(impl->compressed_indices(), impl->plain_indices(), abi_version_of(at_a))) {
    // Recorded before a write to the operand: it can never serve again, and it holds the plan of a
    // transpose autograd.py is about to rebuild. The product that next serves records afresh.
    if (PyDict_DelItem(held, key.ptr()) < 0) PyErr_Clear();
    return nullptr;
  }
  // autograd._transposed_layout(b): b = x^T is the transpose of a contiguous matrix unless it is
  // contiguous itself, which it is when x has one row or one column.
  at::Tensor b = xt.t();
  const bool b_transposed = !b.is_contiguous();
  at::Tensor out;
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
    // autograd._in_layout(result, b_transposed)
    out = b_transposed ? planned_matmul_dense_transpose(s, *served).t() : std::move(*served);
  }
  if (!grad) return THPVariable_Wrap(out.t());
  // torch's values(), sharing `a`'s version counter, as PlannedMatmulGrad saves it.
  at::Tensor values = at_a.values();
  at::Tensor recorded = PlannedMatmulGradT::apply(
      values, b, PlannedMatmulProduct{at_a, std::move(out)}, b_transposed);
  return THPVariable_Wrap(recorded.t());
}

inline PyObject* planned_matmul_call(PyObject* /*module*/, PyObject* const* args,
                                     Py_ssize_t nargsf, PyObject* kwnames) {
  HANDLE_TH_ERRORS
  const PlannedMatmul& s = planned_matmul_state();
  if (PyVectorcall_NARGS(nargsf) == 2 && kwnames == nullptr &&
      reinterpret_cast<PyObject*>(Py_TYPE(args[1])) == s.tensor_type &&
      PyList_CheckExact(s.enabled) && PyList_GET_SIZE(s.enabled) >= 1 &&
      PyList_GET_ITEM(s.enabled, 0) == Py_True) {
    const bool b_grad = planned_matmul_wants_grad(THPVariable_Unpack(args[1]));
    if (reinterpret_cast<PyObject*>(Py_TYPE(args[0])) == s.stensor_type) {
      if (!b_grad) {
        PyObject* served = planned_matmul_try(s, args[0], args[1]);
        if (served != nullptr || PyErr_Occurred()) return served;
      }
    } else if (planned_matmul_torch_csr(s, args[0])) {
      if (!b_grad && !planned_matmul_wants_grad(THPVariable_Unpack(args[0])))
        return planned_matmul_torch_call(s, args[0], args[1], args, nargsf);
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
                                         py::object torch_operands, py::object operand_ref_type,
                                         py::object operand_died, const std::string& seen_attr,
                                         Py_ssize_t max_plans, const std::string& probed_kw,
                                         bool match_host_threads, bool atparallel,
                                         py::object module, const std::string& doc) {
  if (!PyCallable_Check(ordinary.ptr())) throw py::type_error("ordinary must be callable");
  if (!PyType_Check(stensor_type.ptr()) || !PyType_Check(tensor_type.ptr()) ||
      !PyType_Check(operand_ref_type.ptr()))
    throw py::type_error("stensor_type, tensor_type and operand_ref_type must be types");
  if (!PyList_CheckExact(generation.ptr()) || PyList_GET_SIZE(generation.ptr()) < 1 ||
      !PyList_CheckExact(enabled.ptr()) || PyList_GET_SIZE(enabled.ptr()) < 1)
    throw py::type_error("generation and enabled must be one-element lists");
  if (!PyDict_CheckExact(torch_operands.ptr()))
    throw py::type_error("torch_operands must be a dict");
  if (!PyCallable_Check(operand_died.ptr())) throw py::type_error("operand_died must be callable");
  PlannedMatmul& s = planned_matmul_state();
  planned_matmul_set(s.ordinary, ordinary.ptr());
  planned_matmul_set(s.stensor_type, stensor_type.ptr());
  planned_matmul_set(s.tensor_type, tensor_type.ptr());
  planned_matmul_set(s.generation, generation.ptr());
  planned_matmul_set(s.enabled, enabled.ptr());
  planned_matmul_set(s.torch_operands, torch_operands.ptr());
  planned_matmul_set(s.operand_ref_type, operand_ref_type.ptr());
  planned_matmul_set(s.operand_died, operand_died.ptr());
  Py_XSETREF(s.plans_attr, PyUnicode_InternFromString(plans_attr.c_str()));
  Py_XSETREF(s.seen_attr, PyUnicode_InternFromString(seen_attr.c_str()));
  Py_XSETREF(s.probed_kwnames, Py_BuildValue("(s)", probed_kw.c_str()));
  if (s.storage_attr == nullptr) s.storage_attr = PyUnicode_InternFromString("_storage");
  if (s.value_attr == nullptr) s.value_attr = PyUnicode_InternFromString("_value");
  if (s.key_attr == nullptr) s.key_attr = PyUnicode_InternFromString("key");
  if (s.plans_attr == nullptr || s.seen_attr == nullptr || s.probed_kwnames == nullptr ||
      s.storage_attr == nullptr || s.value_attr == nullptr || s.key_attr == nullptr)
    throw py::error_already_set();
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

}  // namespace scorch_native
